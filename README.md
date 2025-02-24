# PlainQAFact

## News
- Our PlainFact dataset can be downloaded here: [`PlainFact`](https://drive.google.com/drive/folders/1mbb06BbZWogweoxc1I5AE7I7m13qhiRL?usp=sharing), including sentence-level and summary-level granularities.
    - Target_Sentence: The plain language sentence/summary.
    - Original_Abstract: The scientific abstract corresponding to each sentence/summary.
    - External: Whether the sentence includes information does not explicitly present in the scientific abstract. ('yes': explanation, 'no': simplification)
- Our fine-tuned Question Generation model is available on 🤗 Hugging Face: [`QG model`](https://huggingface.co/uzw/bart-large-question-generation) (or download it [here](https://drive.google.com/file/d/1-MA9dfOtCm38yTfiQN9Xm8sRvcRD_Cmc/view?usp=drive_link))


## Overall Framework
<div align="center">
  <img src="https://github.com/zhiwenyou103/PlainQAFact/blob/main/pics/system.jpg" height="500" width="700">
</div>


## Installation
Follow the instructions in [`MedRAG`](https://github.com/Teddy-XiongGZ/MedRAG?tab=readme-ov-file#requirements) to install PyTorch and the required packages.
Then, run the following command:
```bash
pip install -r requirements.txt
```
