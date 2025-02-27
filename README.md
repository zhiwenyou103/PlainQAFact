# PlainQAFact

## News
- Our PlainFact dataset can be downloaded here: [`PlainFact`](https://drive.google.com/drive/folders/1mbb06BbZWogweoxc1I5AE7I7m13qhiRL?usp=sharing), including sentence-level and summary-level granularities.
    - Target_Sentence: The plain language sentence/summary.
    - Original_Abstract: The scientific abstract corresponding to each sentence/summary.
    - External: Whether the sentence includes information does not explicitly present in the scientific abstract. ('yes': explanation, 'no': simplification)
- Our fine-tuned Question Generation model is available on 🤗 Hugging Face: [`QG model`](https://huggingface.co/uzw/bart-large-question-generation) (or download it [here](https://drive.google.com/file/d/1-MA9dfOtCm38yTfiQN9Xm8sRvcRD_Cmc/view?usp=drive_link))

> NOTE: This repo is heavily relied on [QAFactEval](https://github.com/salesforce/QAFactEval), [QAEval](https://github.com/danieldeutsch/qaeval), and [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG).


## Overall Framework
<div align="center">
  <img src="https://github.com/zhiwenyou103/PlainQAFact/blob/main/pics/system.jpg" height="500" width="750">
</div>


## Installation
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

## Model Downloading
In PlainQAFact, we use [`Llama 3.1 8B Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for answer extraction, fine-tuned [`QG model`](https://huggingface.co/uzw/bart-large-question-generation), and the original question answering model from [QAFactEval](https://github.com/salesforce/QAFactEval).

Download the pre-trained QA model and our pre-trained classifier through `bash download_question_answering.sh`.


## Usage of PlainQAFact
### Running through our PlainFact dataset
Before running the following command, please download the question answering and learned classifier models through above instructions. 
```bash
python3 run.py \
    --classifier_type learned \  # Options: 'learned', 'llama', 'gpt'
    --input_file data/summary_level.csv \ # path of the input dataset 
    --classifier_path path/to/learned_classifier \ # path of the classifier
    --llm_model_path meta-llama/Llama-3.1-8B-Instruct \ # path of the answer extractor
    --question_generation_model_path uzw/bart-large-question-generation \ # path of the question generation model
    --qa_answering_model_dir models/answering \ # path of the question answering model
    --knowledge_base combined \ # knowledge bases for retrieval
    --answer_selection_strategy llm-keywords  # Options: 'llm-keywords', 'gpt-keywords', 'none'
```

### Running through your own data
Please modify the [`default_config.py`](https://github.com/zhiwenyou103/PlainQAFact/blob/main/default_config.py#L17) file `Line 17-19` to indicate the heading/key names of your dataset. We currently support `.json`, `.txt`, and `.csv` file.
```bash
python3 run.py \
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
