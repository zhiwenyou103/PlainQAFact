from dataclasses import dataclass

@dataclass
class Config:
    classifier_type: str = 'learned'  # Options: 'learned', 'llama', 'gpt'
    classifier_path: str = '/path/to/classifier'
    question_generation_model_path: str = '/path/to/qa_generation'
    qa_answering_model_dir: str = '/path/to/qa_answering'
    llm_model_path: str = '/path/to/llm_model'


    input_file: str = '' # Supports .json, .csv, or .txt files
    delimiter: str = ','
    encoding: str = 'utf-8'

    ### ===Modify the heading/key names of your own PLS and abstracts=== ###
    target_sentence_col: str = 'Target_Sentence' # target summary/sentence
    abstract_col: str = 'Original_Abstract' # original abstract (source)
    input_file_format: str = 'csv'


    retrieval_k: int = 3 # Number of the retrieved snippets for each query
    knowledge_base: str = 'combined'  # Options: 'textbooks', 'statpearls', 'combined'

    answer_selection_strategy: str = 'llm-keywords'  # Options: 'llm-keywords', 'gpt-keywords', 'none'
    openai_api_key: str = 'your openai api key'

    cuda_device: int = 0
    use_bertscore: bool = True
    verbose: bool = True
    generation_batch_size: int = 1
    answering_batch_size: int = 1
    scoring_batch_size: int = 1