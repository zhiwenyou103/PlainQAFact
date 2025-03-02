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

        super().__init__(**qaeval_params)
        
        bertscore_scorer = BertScoreScorer(cuda_device=cuda_device, batch_size=scoring_batch_size)
        self.scorer.scorers.append(bertscore_scorer)

    def _initialize_classifier(self):
        if self.classifier_type == 'learned':
            return LearnedClassifier(self.classifier_path, device=self.cuda_device)
        elif self.classifier_type == 'llama':
            return LLMClassifier('llama', model_path=self.classifier_path, device=self.cuda_device)
        elif self.classifier_type == 'gpt':
            return LLMClassifier('gpt', openai_key=self.openai_api_key)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

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
        else:  # combined
            textbook_knowledge = self._retrieve_knowledge(summaries, "Textbooks")
            statpearl_knowledge = self._retrieve_knowledge(summaries, "StatPearls")
            return [f"{abs_text} {tb_know} {sp_know}"
                    for abs_text, tb_know, sp_know in zip(abstracts, textbook_knowledge, statpearl_knowledge)]
    
    def _initialize_answer_extractor(self):
        if self.answer_selection_strategy == 'llm-keywords':
            return pipeline(
                "text-generation",
                model=self.llm_model_path,
                device=self.cuda_device
            )
        elif self.answer_selection_strategy == 'gpt-keywords':
            return OpenAI(api_key=self.openai_api_key)
        else:
            return None

    def evaluate(self, target_sentences: List[str], abstracts: List[str]) -> Dict:
        if len(target_sentences) != len(abstracts):
            raise ValueError("The number of target sentences must match the number of abstracts")

        classifier = self._initialize_classifier()
        external, external_abs = [], []
        internal, internal_abs = [], []
        
        for abstract, sentence in zip(abstracts, target_sentences):
            label, _ = classifier.predict(abstract, sentence)
            if label == 'yes':
                external.append(sentence)
                external_abs.append(abstract)
            else:
                internal.append(sentence)
                internal_abs.append(abstract)

        external_summaries = [[s] for s in external]
        internal_summaries = [[s] for s in internal]

        try:
            generator = self._initialize_answer_extractor()
        except Exception as e:
            print(f"Error initializing generator: {str(e)}")
            sys.exit(1)

        if external:
            combined_contexts = self._get_combined_knowledge(external_abs, external_summaries)
            external_results = self.score_batch_qafacteval(
                combined_contexts,
                external_summaries,
                return_qa_pairs=True,
                generator=generator
            )
        else:
            external_results = []

        if internal:
            internal_results = self.score_batch_qafacteval(
                internal_abs,
                internal_summaries,
                return_qa_pairs=True,
                generator=generator
            )
        else:
            internal_results = []

        external_scores = [metrics['qa-eval']['bertscore'] for metrics, *_ in external_results] if external_results else []
        internal_scores = [metrics['qa-eval']['bertscore'] for metrics, *_ in internal_results] if internal_results else []

        return {
            'external_results': external_results,
            'internal_results': internal_results,
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


        classifier = self._initialize_classifier()
        external, external_abs = [], []
        internal, internal_abs = [], []
        
        for abstract, sentence in zip(df[self.abstract_col], df[self.target_sentence_col]):
            label, _ = classifier.predict(abstract, sentence)
            if label == 'yes':
                external.append(sentence)
                external_abs.append(abstract)
            else:
                internal.append(sentence)
                internal_abs.append(abstract)

        external_summaries = [[s] for s in external]
        internal_summaries = [[s] for s in internal]

        try:
            generator = self._initialize_answer_extractor()
        except Exception as e:
            print(f"Error initializing generator: {str(e)}")
            sys.exit(1)

        if external:
            combined_contexts = self._get_combined_knowledge(external_abs, external_summaries)
            external_results = self.score_batch_qafacteval(
                combined_contexts,
                external_summaries,
                return_qa_pairs=True,
                generator=generator
            )
        else:
            external_results = []

        if internal:
            internal_results = self.score_batch_qafacteval(
                internal_abs,
                internal_summaries,
                return_qa_pairs=True,
                generator=generator
            )
        else:
            internal_results = []

        external_scores = [metrics['qa-eval']['bertscore'] for metrics, *_ in external_results] if external_results else []
        internal_scores = [metrics['qa-eval']['bertscore'] for metrics, *_ in internal_results] if internal_results else []

        return {
            'external_results': external_results,
            'internal_results': internal_results,
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
        
        if self.answer_selection_strategy == 'llm-keywords' or self.answer_selection_strategy == 'gpt-keywords':
            generator = generator
        else:
            generator = None

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
