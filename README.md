<div align="left">
  <a href="https://pypi.org/project/plainqafact/"><img src="https://img.shields.io/badge/PyPI-library-blue" alt="PyPI"></a>
  <a href="https://arxiv.org/abs/2503.08890"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/uzw/PlainFact"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?color=8A2BE2" alt="Dataset"></a>
</div>


# PlainQAFact
`PlainQAFact` is a retrieval-augmented and question-answering (QA)-based factuality evaluation framework for assessing the factuality of biomedical plain language summarization tasks. `PlainFact` is a high-quality human-annotated dataset with fine-grained explanation (i.e., added information) annotations.

## News
- (2025.03.11) `PlainFact` is now available on 🤗 Hugging Face: [PlainFact](https://huggingface.co/datasets/uzw/PlainFact) for sentence-level data and [PlainFact-summary](https://huggingface.co/datasets/uzw/PlainFact-summary) for summary-level data.
- (2025.03.02) Pre-embedded vector bases of [Textbooks](https://github.com/jind11/MedQA) and [StatPearls](https://www.statpearls.com/) can be downloaded [here](https://drive.google.com/file/d/1DVJoKwuFLCIs0iEvrkP05KS9y6Z6kjXT/view?usp=sharing).
- (2025.03.01) 🚨🚨🚨 `PlainQAFact` is now on [PyPI](https://pypi.org/project/plainqafact/)! Simply use `pip install plainqafact` to load our pipeline!
- (2025.02.24) Our `PlainFact` dataset can be downloaded here: [`PlainFact`](https://drive.google.com/drive/folders/1mbb06BbZWogweoxc1I5AE7I7m13qhiRL?usp=sharing), including sentence-level and summary-level granularities.
    - Target_Sentence: The plain language sentence/summary.
    - Original_Abstract: The scientific abstract corresponding to each sentence/summary.
    - External: Whether the sentence includes information does not explicitly present in the scientific abstract. (`yes`: explanation, `no`: simplification)
    - _We will release the full version of PlainFact soon (including Category and Relation information). Stay tuned!_
- (2025.02.24) Our fine-tuned Question Generation model is available on 🤗 Hugging Face: [`QG model`](https://huggingface.co/uzw/bart-large-question-generation) (or download it [here](https://drive.google.com/file/d/1-MA9dfOtCm38yTfiQN9Xm8sRvcRD_Cmc/view?usp=drive_link))

> NOTE: This repo is heavily relied on [QAFactEval](https://github.com/salesforce/QAFactEval), [QAEval](https://github.com/danieldeutsch/qaeval), and [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG).


## Overall Framework
<div align="center">
  <img src="https://github.com/zhiwenyou103/PlainQAFact/blob/main/pics/plainqafact.jpg" height="550" width="750">
</div>

## Model Downloading
In PlainQAFact, we use [`pre-trained classifier`](https://drive.google.com/file/d/1PuQA6bYsnKrIUU1i3ioKhr7xxWnJhdSh/view?usp=sharing) to distinguish simplification and explanation sentences, [`Llama 3.1 8B Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for answer extraction, fine-tuned [`QG model`](https://huggingface.co/uzw/bart-large-question-generation), and the original question answering model from [QAFactEval](https://github.com/salesforce/QAFactEval).

Download the **pre-trained QA model** and our **pre-trained classifier** through `bash download_question_answering.sh`.

## Quickstart
```bash
conda create -n plainqafact python=3.9
pip install plainqafact
```

After installation, make sure you initialized `git-lfs` as required by [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG). Then, you can directly use PlainQAFact through:
```python
from plainqafact import PlainQAFact

metric = PlainQAFact(
    cuda_device=0,
    classifier_type='learned',
    classifier_path='models/learned_classifier',
    llm_model_path='meta-llama/Llama-3.1-8B-Instruct',
    question_generation_model_path='uzw/bart-large-question-generation',
    qa_answering_model_dir='models/answering',
    knowledge_base='combined', # retrieve from both Textbooks and StatPearls KBs
    scoring_batch_size=1,
    answer_selection_strategy='llm-keywords'
)

# choice 1: interactively evaluate summaries
# summaries:
target_sentences = [
    "The study shows aspirin reduces heart attack risk.",
    "Patients with high blood pressure should exercise regularly."
]
# scientific abstracts:
abstracts = [
    "A comprehensive clinical trial demonstrated that daily aspirin administration significantly decreased the incidence of myocardial infarction in high-risk patients.",
    "Research indicates that regular physical activity is an effective intervention for managing hypertension in adult patients."
]

results = metric.evaluate(target_sentences, abstracts)

print(f"Explanation score (mean: {results['external_mean']:.4f}):", results['external_scores'])
print(f"Simplification score (mean: {results['internal_mean']:.4f}):", results['internal_scores'])
print(f"PlainQAFact score: {results['overall_mean']:.4f}")
```

Or you can evaluate a data file through:
```python
from plainqafact import PlainQAFact

metric = PlainQAFact(
    cuda_device=0,
    classifier_type='learned',
    classifier_path='models/learned_classifier',
    llm_model_path='meta-llama/Llama-3.1-8B-Instruct',
    question_generation_model_path='uzw/bart-large-question-generation',
    qa_answering_model_dir='models/answering',
    knowledge_base='combined',
    scoring_batch_size=1,
    answer_selection_strategy='llm-keywords',
    target_sentence_col='Target_Sentence', # name of your summary's key (column)
    abstract_col='Original_Abstract', # name of your abstract's key (column)
    input_file_format='csv' # your input file format
)

# choice 2: directly evaluate a data file:
results = metric.evaluate_all(input_file='your_data.csv')

print(f"Explanation score (mean: {results['external_mean']:.4f}):", results['external_scores'])
print(f"Simplification score (mean: {results['internal_mean']:.4f}):", results['internal_scores'])
print(f"PlainQAFact score: {results['overall_mean']:.4f}")
```

## Option 2: Install from source
### Installation
- First, create a new conda env: `conda create -n plainqafact python=3.9` and clone our repo.
- `cd PlainQAFact`
- Follow the instructions in [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG?tab=readme-ov-file#requirements) to install PyTorch and other required packages.
- Then, run the following command:
    ```bash
    conda install git
    pip install -r requirements.txt
    ```
- Finally, install the old tokenizer package through:
    ```bash
    pip install transformers_old_tokenizer-3.1.0-py3-none-any.whl
    ```
    
### Running through our PlainFact dataset
Before running the following command, please download the question answering and learned classifier models through above instructions. 
```bash
python3 run.py \
    --cuda_device 0 \
    --classifier_type learned \  # Options: 'learned', 'llama', 'gpt'
    --input_file data/summary_level.csv \ # path of the input dataset 
    --classifier_path path/to/learned_classifier \ # path of the classifier
    --llm_model_path meta-llama/Llama-3.1-8B-Instruct \ # path of the answer extractor
    --question_generation_model_path uzw/bart-large-question-generation \ # path of the question generation model
    --qa_answering_model_dir models/answering \ # path of the question answering model
    --knowledge_base combined \ # knowledge bases for retrieval, options: textbooks, statpearls, pubmed, wikipedia, combined
    --answer_selection_strategy llm-keywords  # Options: 'llm-keywords', 'gpt-keywords', 'none'
```

### Running through your own data
Please modify the [`default_config.py`](https://github.com/zhiwenyou103/PlainQAFact/blob/main/default_config.py#L17) file `Line 17-19` to indicate the heading/key names of your dataset. We currently support `.json`, `.txt`, and `.csv` file.
```bash
python3 run.py \
    --cuda_device 0 \
    --classifier_type learned \
    --input_file your_own_data.json \
    --input_file_format json \
    --classifier_path path/to/learned_classifier \
    --llm_model_path meta-llama/Llama-3.1-8B-Instruct \
    --question_generation_model_path uzw/bart-large-question-generation \
    --qa_answering_model_dir models/answering \
    --knowledge_base textbooks \
    --answer_selection_strategy llm-keywords
```

### Easily replace the pre-trained classifier to OpenAI models or your own
We provides options to easily replace our pre-trained classisifer tailored for the biomedical plain language summarization tasks to other tasks. You may simply set `--classifier_type` as `gpt` and provide your OpenAI API key in the [`default_config.py`](https://github.com/zhiwenyou103/PlainQAFact/blob/main/default_config.py#L26) file `Line 26` to run PlainQAFact.
```bash
python3 run.py \
    --cuda_device 0 \
    --classifier_type gpt \
    --input_file your_own_data.json \
    --input_file_format json \
    --llm_model_path meta-llama/Llama-3.1-8B-Instruct \
    --question_generation_model_path uzw/bart-large-question-generation \
    --qa_answering_model_dir models/answering \
    --knowledge_base textbooks \
    --answer_selection_strategy llm-keywords
```

### Using other Knowledge Bases for retrieval
Currently, we only experiment with two KBs: Textbooks and StatPearls. You may want to use your customized KBs for more accurate retrieval. In PlainQAFact, we combine both Textbooks and StatPearls and concatenate with the scientific abstracts. Set `--knowledge_base textbooks` as `combined` to reproduce our results.


> NOTE: Using Llama 3.1 8B model for both classification and answer extraction would take over 40 GB GPU memory. We recommend to use our pre-trained classifier or OpenAI models for classification if the GPU memory is limited.


## Citation Information
For the use of PlainQAFact and PlainFact benchmark, please cite:
```
@misc{you2025plainqafactretrievalaugmentedfactualconsistency,
      title={PlainQAFact: Retrieval-augmented Factual Consistency Evaluation Metric for Biomedical Plain Language Summarization}, 
      author={Zhiwen You and Yue Guo},
      year={2025},
      eprint={2503.08890},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.08890}, 
}
```
## Contact Information
If you have any questions, please email `zhiweny2@illinois.edu`.
