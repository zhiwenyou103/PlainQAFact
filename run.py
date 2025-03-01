import argparse
import sys
import os
import pandas as pd
import numpy as np
from transformers import pipeline
from src.medrag import MedRAG
from plainqafact import PlainQAFact
from classifier import LearnedClassifier, LLMClassifier
from default_config import Config
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description='Run medical text evaluation pipeline')
    parser.add_argument('--classifier_path', help='Path to input CSV file')
    parser.add_argument('--input_file', required=True, help='Path to input file (.csv, .json, or .txt)')
    parser.add_argument('--input_file_format', choices=['csv', 'json', 'txt'], default='csv',
                        help='Input file format')
    parser.add_argument('--delimiter', default=',', help='Delimiter for csv/txt files')
    parser.add_argument('--encoding', default='utf-8', help='File encoding (default: utf-8)')
    parser.add_argument('--question_generation_model_path', required=True, help='Path to QA generation model')
    parser.add_argument('--qa_answering_model_dir', required=True, help='Path to QA answering model directory')
    parser.add_argument('--llm_model_path', required=True, help='Path to the Llama model')
    parser.add_argument('--cuda_device', default='cuda:0', help='Define your cuda device (e.g., cuda:0)')
    parser.add_argument('--scoring_batch_size', default=1, help='Batch size of BERTScore evaluation')
    parser.add_argument('--verbose', default=True)
    parser.add_argument(
        '--classifier_type',
        choices=['learned', 'llama', 'gpt'],
        default='learned',
        help='Type of classifier to use (default: learned)'
    )
    parser.add_argument(
        '--knowledge_base',
        choices=['textbooks', 'statpearls', 'combined'],
        default='combined',
        help='Knowledge base to use for retrieval (default: combined)'
    )
    parser.add_argument(
        '--answer_selection_strategy',
        choices=['llm-keywords', 'gpt-keywords', 'none'],
        default='llm-keywords',
        help='Strategy for answer selection (default: llm-keywords)'
    )
    return parser.parse_args()

def initialize_answer_extractor(config: Config):
    if config.answer_selection_strategy == 'llm-keywords':
        return pipeline(
            "text-generation",
            model=config.llm_model_path,
            device=config.cuda_device
        )
    elif config.answer_selection_strategy == 'gpt-keywords':
        return OpenAI(api_key=config.openai_api_key)
    else:
        return None

def initialize_classifier(config: Config):
    if config.classifier_type == 'learned':
        return LearnedClassifier(config.classifier_path, device=config.cuda_device)
    elif config.classifier_type == 'llama':
        return LLMClassifier('llama', model_path=config.classifier_path, device=config.cuda_device)
    elif config.classifier_type == 'gpt':
        return LLMClassifier('gpt', openai_key=config.openai_api_key)
    else:
        raise ValueError(f"Unknown classifier type: {config.classifier_type}")

def process_data(config: Config):
    if config.input_file_format == 'csv':
        df = pd.read_csv(config.input_file, delimiter=config.delimiter, encoding=config.encoding)
    elif config.input_file_format == 'json':
        df = pd.read_json(config.input_file, encoding=config.encoding)
    elif config.input_file_format == 'txt':
        df = pd.read_csv(config.input_file, delimiter=config.delimiter, encoding=config.encoding)
    else:
        raise ValueError(f"Unsupported file format: {config.input_file_format}")

    if config.target_sentence_col not in df.columns or config.abstract_col not in df.columns:
        raise ValueError(f"Required columns '{config.target_sentence_col}' and '{config.abstract_col}' "
                        f"must be present in the input file")

    sentences = df[config.target_sentence_col]
    abstracts = df[config.abstract_col]

    classifier = initialize_classifier(config)
    external, external_abs, internal, internal_abs = [], [], [], []
    for abstract, sentence in zip(abstracts, sentences):
        label, _ = classifier.predict(abstract, sentence)
        if label == 'yes':
            external.append(sentence)
            external_abs.append(abstract)
        else:
            internal.append(sentence)
            internal_abs.append(abstract)

    return prepare_evaluation_data(external, external_abs, internal, internal_abs)

def prepare_evaluation_data(external, external_abs, internal, internal_abs):
    external_pls = [[s] for s in external]
    internal_pls = [[s] for s in internal]
    return {
        'external': {'abstracts': external_abs, 'summaries': external_pls},
        'internal': {'abstracts': internal_abs, 'summaries': internal_pls}
    }

def retrieve_knowledge(queries, config: Config, corpus_name: str):
    knowledge = []
    for query in queries:
        medrag = MedRAG(retrieval=True, retriever_name="MedCPT", corpus_name=corpus_name, corpus_cache=True)
        snippets, _ = medrag.answer(question=query[0], k=config.retrieval_k)
        content = ' '.join([item['content'].strip().replace('\n', '') for item in snippets])
        knowledge.append(content)
    return knowledge

def get_combined_knowledge(abstracts, summaries, config: Config):
    if config.knowledge_base == 'textbooks':
        knowledge = retrieve_knowledge(summaries, config, "Textbooks")
        return [f"{abs_text}{tb_know}"
                for abs_text, tb_know in zip(abstracts, knowledge)]

    elif config.knowledge_base == 'statpearls':
        knowledge = retrieve_knowledge(summaries, config, "StatPearls")
        return [f"{abs_text}{sp_know}"
                for abs_text, sp_know in zip(abstracts, knowledge)]

    else:
        textbook_knowledge = retrieve_knowledge(summaries, config, "Textbooks")
        statpearl_knowledge = retrieve_knowledge(summaries, config, "StatPearls")
        return [f"{abs_text} {tb_know} {sp_know}"
                for abs_text, tb_know, sp_know in zip(abstracts, textbook_knowledge, statpearl_knowledge)]

def main():
    args = parse_args()
    config = Config()
    for key, value in vars(args).items():
        setattr(config, key, value)

    data = process_data(config)

    combined_contexts = get_combined_knowledge(
        data['external']['abstracts'],
        data['external']['summaries'],
        config
    )

    metric = PlainQAFact(
        classifier_type=config.classifier_type,
        classifier_path=config.classifier_path,
        generation_model_path=config.question_generation_model_path,
        answering_model_dir=config.qa_answering_model_dir,
        cuda_device=config.cuda_device,
        knowledge_base=config.knowledge_base,
        generation_batch_size=config.generation_batch_size,
        answering_batch_size=config.answering_batch_size,
        scoring_batch_size=config.scoring_batch_size,
        answer_selection_strategy=config.answer_selection_strategy
    )

    try:
        generator = initialize_answer_extractor(config)
    except Exception as e:
        print(f"Error initializing generator: {str(e)}")
        sys.exit(1)

    # Run evaluation
    external_results = metric.score_batch_qafacteval(
        combined_contexts,
        data['external']['summaries'],
        return_qa_pairs=True,
        generator=generator
    )

    internal_results = metric.score_batch_qafacteval(
        data['internal']['abstracts'],
        data['internal']['summaries'],
        return_qa_pairs=True,
        generator=generator
    )

    external_scores = [metrics['qa-eval']['bertscore'] for metrics, *_ in external_results]
    internal_scores = [metrics['qa-eval']['bertscore'] for metrics, *_ in internal_results]

    print(f"Explanation scores (mean: {np.mean(external_scores):.4f}):", external_scores)
    print(f"Simplification scores (mean: {np.mean(internal_scores):.4f}):", internal_scores)
    print(f"PlainQAFact score: {np.mean(external_scores + internal_scores):.4f}")


if __name__ == "__main__":
    main()
