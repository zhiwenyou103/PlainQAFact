from typing import Dict, List, Union
import sys
from transformers.data.metrics.squad_metrics import compute_f1
from qaeval.metric import QAEval
import bert_score
from transformers import pipeline
from bertscore_scorer import BertScoreScorer
import pandas as pd
import json
import numpy as np
from classifier import LearnedClassifier, LLMClassifier
from openai import OpenAI
from src.medrag import MedRAG
import torch
import gc
import nltk
nltk.download('punkt')

MetricsDict = Dict[str, float]
SummaryType = Union[str, List[str]]

def get_filter(qa_summ, answers_ref):
    answerable = []
    a_orig = []
    for qa_summ_ in qa_summ:
        answerability = (qa_summ_[1] > qa_summ_[2])
        answerable.append(answerability)
        a_orig.append(qa_summ_[0])

    f1s = [compute_f1(answer, prediction) for \
        answer, prediction in zip(answers_ref, a_orig)]
    bool_f1 = [x > 0.60 for x in f1s]
    bool_total = [x and y for x, y in zip(bool_f1, answerable)]
    return bool_total

class PlainQAFact(QAEval):
    def __init__(
            self,
            cuda_device: int,
            scoring_batch_size: int,
            answer_selection_strategy: str,
            openai_api_key: str = '',
            classifier_type: str = None,
            classifier_path: str = None,
            llm_model_path: str = None,
            question_generation_model_path: str = None,
            qa_answering_model_dir: str = None,
            knowledge_base: str = None,
            target_sentence_col: str = 'Target_Sentence',
            abstract_col: str = 'Original_Abstract',
            input_file_format: str = 'csv',
            delimiter: str = ',',
            encoding: str = 'utf-8',
            retrieval_k: int = 3,
            generation_batch_size: int = 1,
            answering_batch_size: int = 1,
            verbose: bool = True,
            *args,
            **kwargs):

        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        self.cuda_device = cuda_device
        self.classifier_type = classifier_type
        self.classifier_path = classifier_path
        self.llm_model_path = llm_model_path
        self.question_generation_model_path = question_generation_model_path
        self.qa_answering_model_dir = qa_answering_model_dir
        self.knowledge_base = knowledge_base
        self.answer_selection_strategy = answer_selection_strategy
        self.openai_api_key = openai_api_key
        self.target_sentence_col = target_sentence_col
        self.abstract_col = abstract_col
        self.input_file_format = input_file_format
        self.delimiter = delimiter
        self.encoding = encoding
        self.retrieval_k = retrieval_k
        self.generation_batch_size = generation_batch_size
        self.answering_batch_size = answering_batch_size
        self.scoring_batch_size = scoring_batch_size
            
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")

        qaeval_params = {
            'cuda_device': cuda_device,
            'generation_model_path': question_generation_model_path,
            'answering_model_dir': qa_answering_model_dir, 
            'generation_batch_size': generation_batch_size,
            'answering_batch_size': answering_batch_size, 
            'verbose': verbose
        }
        
        plainqafact_params = [
            'classifier_type', 'classifier_path', 'llm_model_path',
            'question_generation_model_path', 'qa_answering_model_dir',
            'knowledge_base', 'target_sentence_col', 'abstract_col',
            'input_file_format', 'delimiter', 'encoding', 'retrieval_k',
            'generation_batch_size', 'answering_batch_size', 'openai_api_key',
            'scoring_batch_size', 'answer_selection_strategy'
        ]
        
        for key, value in kwargs.items():
            if key not in plainqafact_params:
                qaeval_params[key] = value

        self._answer_extractor = None
        self._classifier = None
        self._qa_model = None
        self._bertscore_scorer = None

        self._prevent_qa_model_load = True
        super().__init__(**qaeval_params)
        self._prevent_qa_model_load = False

    def __del__(self):
        try:
            if hasattr(self, '_classifier') and self._classifier is not None:
                del self._classifier
            if hasattr(self, '_qa_model') and self._qa_model is not None:
                del self._qa_model
            if hasattr(self, '_answer_extractor') and self._answer_extractor is not None:
                del self._answer_extractor
            if hasattr(self, '_bertscore_scorer') and self._bertscore_scorer is not None:
                del self._bertscore_scorer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")

    def _initialize_answer_extractor(self):
        if self._answer_extractor is None:
            try:
                if self.answer_selection_strategy == 'llm-keywords':
                    model_kwargs = {
                        "device_map": "auto",
                        "torch_dtype": torch.float16,
                        "low_cpu_mem_usage": True
                    }
                    self._answer_extractor = pipeline(
                        "text-generation",
                        model=self.llm_model_path,
                        model_kwargs=model_kwargs
                    )
                elif self.answer_selection_strategy == 'gpt-keywords':
                    self._answer_extractor = OpenAI(api_key=self.openai_api_key)
                else:
                    self._answer_extractor = None
            except Exception as e:
                print(f"Warning: Failed to initialize answer extractor: {str(e)}")
                self._answer_extractor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
        return self._answer_extractor

    def _initialize_bertscore_scorer(self):
        if self._bertscore_scorer is None:
            self._bertscore_scorer = BertScoreScorer(cuda_device=self.cuda_device, batch_size=self.scoring_batch_size)
            self.scorer.scorers.append(self._bertscore_scorer)
        return self._bertscore_scorer

    def _initialize_classifier(self):
        if self._classifier is None:
            if self.classifier_type == 'learned':
                self._classifier = LearnedClassifier(self.classifier_path, device=self.cuda_device if torch.cuda.is_available() else "cpu")
            elif self.classifier_type == 'llama':
                self._classifier = LLMClassifier('llama', model_path=self.classifier_path, device=self.cuda_device if torch.cuda.is_available() else "cpu")
            elif self.classifier_type == 'gpt':
                self._classifier = LLMClassifier('gpt', openai_key=self.openai_api_key)
            else:
                raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        return self._classifier

    def _retrieve_knowledge(self, queries, corpus_name: str):
        knowledge = []
        for query in queries:
            medrag = MedRAG(cuda_device=self.cuda_device, retrieval=True, retriever_name="MedCPT", corpus_name=corpus_name, corpus_cache=True)
            snippets, _ = medrag.answer(question=query[0], k=self.retrieval_k)
            content = ' '.join([item['content'].strip().replace('\n', '') for item in snippets])
            knowledge.append(content)
        return knowledge

    def _get_combined_knowledge(self, abstracts, summaries):
        if self.knowledge_base == 'textbooks':
            knowledge = self._retrieve_knowledge(summaries, "Textbooks")
            return [f"{abs_text}{tb_know}"
                    for abs_text, tb_know in zip(abstracts, knowledge)]
        elif self.knowledge_base == 'statpearls':
            knowledge = self._retrieve_knowledge(summaries, "StatPearls")
            return [f"{abs_text}{sp_know}"
                    for abs_text, sp_know in zip(abstracts, knowledge)]
        elif self.knowledge_base == 'pubmed':
            knowledge = self._retrieve_knowledge(summaries, "PubMed")
            return [f"{abs_text}{sp_know}"
                    for abs_text, sp_know in zip(abstracts, knowledge)]
        elif self.knowledge_base == 'wikipedia':
            knowledge = self._retrieve_knowledge(summaries, "Wikipedia")
            return [f"{abs_text}{sp_know}"
                    for abs_text, sp_know in zip(abstracts, knowledge)]
        else:  # combined
            textbook_knowledge = self._retrieve_knowledge(summaries, "Textbooks")
            statpearl_knowledge = self._retrieve_knowledge(summaries, "StatPearls")
            return [f"{abs_text} {tb_know} {sp_know}"
                    for abs_text, tb_know, sp_know in zip(abstracts, textbook_knowledge, statpearl_knowledge)]

    def evaluate(self, summaries: List[str], abstracts: List[str]) -> Dict:
        if len(summaries) != len(abstracts):
            raise ValueError("The number of summaries must match the number of abstracts")

        classifier = self._initialize_classifier()
        self._initialize_bertscore_scorer()

        all_sentences = []
        all_abs = []
        for summary, abstract in zip(summaries, abstracts):
            sentences = nltk.sent_tokenize(summary)
            all_sentences.append(sentences)
            all_abs.append([abstract] * len(sentences))

        external, external_abs = [], []
        internal, internal_abs = [], []

        for sent_list, abs_list in zip(all_sentences, all_abs):
            temp_external, temp_external_abs = [], []
            temp_internal, temp_internal_abs = [], []
            for sentence, abstract in zip(sent_list, abs_list):
                label, _ = classifier.predict(abstract, sentence)
                if label == 'yes':
                    temp_external.append([sentence])
                    temp_external_abs.append(abstract)
                else:
                    temp_internal.append([sentence])
                    temp_internal_abs.append(abstract)
            external.append(temp_external)
            external_abs.append(temp_external_abs)
            internal.append(temp_internal)
            internal_abs.append(temp_internal_abs)

        external_summaries = external
        internal_summaries = internal

        generator = None
        try:
            generator = self._initialize_answer_extractor()
            if generator is None and self.answer_selection_strategy == 'llm-keywords':
                print("Warning: Failed to initialize generator for llm-keywords strategy. Falling back to default strategy.")
                self.answer_selection_strategy = 'none'
        except Exception as e:
            print(f"Error initializing generator: {str(e)}")
            print("Falling back to default strategy.")
            self.answer_selection_strategy = 'none'

        all_external_results = []
        for abs_list, pls_list in zip(external_abs, external_summaries):
            if pls_list:
                combined_contexts = self._get_combined_knowledge(abs_list, pls_list)
                results = self.score_batch_qafacteval(
                    combined_contexts, pls_list, return_qa_pairs=True, generator=generator
                )
                all_external_results.append(results)
            else:
                all_external_results.append([])

        all_internal_results = []
        for abs_list, pls_list in zip(internal_abs, internal_summaries):
            if pls_list:
                results = self.score_batch_qafacteval(
                    abs_list, pls_list, return_qa_pairs=True, generator=generator
                )
                all_internal_results.append(results)
            else:
                all_internal_results.append([])

        external_scores = []
        for results in all_external_results:
            for metrics, *_ in results:
                external_scores.append(metrics['qa-eval']['bertscore'])

        internal_scores = []
        for results in all_internal_results:
            for metrics, *_ in results:
                internal_scores.append(metrics['qa-eval']['bertscore'])

        return {
            'external_results': all_external_results,
            'internal_results': all_internal_results,
            'external_scores': external_scores,
            'internal_scores': internal_scores,
            'external_mean': np.mean(external_scores) if external_scores else 0,
            'internal_mean': np.mean(internal_scores) if internal_scores else 0,
            'overall_mean': np.mean(external_scores + internal_scores) if (external_scores or internal_scores) else 0
        }

    def evaluate_all(self, input_file: str) -> Dict:
        if self.input_file_format == 'csv':
            df = pd.read_csv(input_file, delimiter=self.delimiter, encoding=self.encoding)
        elif self.input_file_format == 'json':
            with open(input_file, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif self.input_file_format == 'txt':
            df = pd.read_csv(input_file, delimiter=self.delimiter, encoding=self.encoding)
        else:
            raise ValueError(f"Unsupported file format: {self.input_file_format}")

        pls_summaries = df[self.target_sentence_col].tolist()
        original_abstracts = df[self.abstract_col].tolist()
        
        pls_sentences = []
        abstracts = []
        for pls, abs in zip(pls_summaries, original_abstracts):
            document = pls
            split_sentences = nltk.sent_tokenize(document)
            pls_sentences.append(split_sentences)
            temp_abs = []
            for sentence in split_sentences:
                if sentence:
                    temp_abs.append(abs)
            abstracts.append(temp_abs)
        
        classifier = self._initialize_classifier()
        self._initialize_bertscore_scorer()
        
        external, external_abs = [], []
        internal, internal_abs = [], []
        
        # for abstract, sentence in zip(df[self.abstract_col], df[self.target_sentence_col]):
        for parent_abstract, parent_sentence in zip(abstracts, pls_sentences):
            temp_external, temp_external_abs = [], []
            temp_internal, temp_internal_abs = [], []
            for sentence, abstract in zip(parent_sentence, parent_abstract):
                label, _ = classifier.predict(abstract, sentence)
                if label == 'yes':
                    temp_external.append([sentence])
                    temp_external_abs.append(abstract)
                else:
                    temp_internal.append([sentence])
                    temp_internal_abs.append(abstract)
            external.append(temp_external)
            external_abs.append(temp_external_abs)
            internal.append(temp_internal)
            internal_abs.append(temp_internal_abs)

        external_summaries = external
        internal_summaries = internal

        generator = None
        try:
            generator = self._initialize_answer_extractor()
            if generator is None and self.answer_selection_strategy == 'llm-keywords':
                print("Warning: Failed to initialize generator for llm-keywords strategy. Falling back to default strategy.")
                self.answer_selection_strategy = 'none'
        except Exception as e:
            print(f"Error initializing generator: {str(e)}")
            print("Falling back to default strategy.")
            self.answer_selection_strategy = 'none'

        all_external_results = []
        for abs_list, pls_list in zip(external_abs, external_summaries):
            if pls_list:
                combined_contexts = self._get_combined_knowledge(abs_list, pls_list)
                results = self.score_batch_qafacteval(
                    combined_contexts, pls_list, return_qa_pairs=True, generator=generator
                )
                all_external_results.append(results)
            else:
                all_external_results.append([])

        all_internal_results = []
        for abs_list, pls_list in zip(internal_abs, internal_summaries):
            if pls_list:
                results = self.score_batch_qafacteval(
                    abs_list, pls_list, return_qa_pairs=True, generator=generator
                )
                all_internal_results.append(results)
            else:
                all_internal_results.append([])

        external_scores = []
        for results in all_external_results:
            for metrics, *_ in results:
                external_scores.append(metrics['qa-eval']['bertscore'])

        internal_scores = []
        for results in all_internal_results:
            for metrics, *_ in results:
                internal_scores.append(metrics['qa-eval']['bertscore'])

        return {
            'external_results': all_external_results,
            'internal_results': all_internal_results,
            'external_scores': external_scores,
            'internal_scores': internal_scores,
            'external_mean': np.mean(external_scores) if external_scores else 0,
            'internal_mean': np.mean(internal_scores) if internal_scores else 0,
            'overall_mean': np.mean(external_scores + internal_scores) if (external_scores or internal_scores) else 0
        }

    def score_batch_qafacteval(
        self,
        source: List[SummaryType],
        summaries: List[List[SummaryType]],
        qa_pairs_precomputed: List = None,
        predictions_lists: List = None,
        return_qa_pairs: bool = False,
        generator=None,
            ) -> List[List[MetricsDict]]:
        
        self._initialize_bertscore_scorer()
        
        if generator is None:
            generator = self._initialize_answer_extractor()
        
        if self.answer_selection_strategy == 'llm-keywords' and generator is None:
            raise ValueError("Generator is required for llm-keywords strategy but could not be initialized")

        source = self._flatten_summaries(source)
        summaries = self._flatten_references_list(summaries)

        (
            source,
            summaries,
            is_empty_list,
        ) = self._get_empty_summary_mask(source, summaries)

        if qa_pairs_precomputed:
            qa_pairs_lists = qa_pairs_precomputed
        else:
            qa_pairs_lists = self._generate_qa_pairs(summaries, self.answer_selection_strategy, generator)

        summaries_cons = [x[0] for x in summaries]
        predictions_lists_consistency = self._answer_questions(summaries_cons, qa_pairs_lists)
        qa_pairs_lists_cons = []
        for x, cur_qa_pair in zip(predictions_lists_consistency, qa_pairs_lists):
            qa_summ_new = [[x["prediction"], x["probability"], \
                x["null_probability"]] for x in x[0]]
            answers_ref = [x["answer"] for x in cur_qa_pair[0]]

            bool_total = get_filter(qa_summ_new, answers_ref)

            cur_qa_pair_keep = [x for count, x in enumerate(cur_qa_pair[0]) if bool_total[count]]
            if not cur_qa_pair_keep:
                cur_qa_pair_keep = []
            qa_pairs_lists_cons.append([cur_qa_pair_keep])

        if predictions_lists:
            predictions_lists = predictions_lists
        else:
            predictions_lists = self._answer_questions(source, qa_pairs_lists_cons)
        
        metrics_list, scores_lists = self._score_predictions(
            source, qa_pairs_lists_cons, predictions_lists
        )

        if return_qa_pairs:
            output = self._combine_outputs(
                metrics_list, qa_pairs_lists_cons, predictions_lists, scores_lists
            )
        else:
            output = metrics_list

        output = self._insert_empty_outputs(output, is_empty_list, return_qa_pairs)
        output_final = []
        for out, qa_pairs_list, predictions_cons in \
                zip(output, qa_pairs_lists, predictions_lists_consistency):
            output_final.append((out[0], out[1], qa_pairs_list, predictions_cons))
        return output_final
