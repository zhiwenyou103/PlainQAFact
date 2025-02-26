import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from transformers import pipeline
from openai import OpenAI

class LearnedClassifier:
    def __init__(self, model_path, device):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, abstract, sentence, return_probability=False):
        encoding = self.tokenizer.encode_plus(
            text=abstract,
            text_pair=sentence,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

            predicted_class = torch.argmax(probabilities, dim=1).item()
            probabilities = probabilities[0].cpu().numpy()

        label = 'yes' if predicted_class == 1 else 'no'

        if return_probability:
            return label, {'no': probabilities[0], 'yes': probabilities[1]}
        else:
            confidence = probabilities[predicted_class]
            return label, confidence


class LLMClassifier:
    def __init__(self, classifier_type: str, model_path: str = None, device: int = None, openai_key: str = None):
        self.max_tokens = 512
        self.classifier_type = classifier_type

        if classifier_type == 'llama':
            self.generator = pipeline(
                "text-generation",
                model=model_path,
                device=device
            )
        elif classifier_type == 'gpt':
            self.client = OpenAI(api_key=openai_key)

    def predict(self, abstract: str, sentence: str) -> tuple:
        if self.classifier_type == 'llama':
            label = self.llama_classifier(sentence, abstract)
        else:
            label = self.gpt_classifier(sentence, abstract)

        confidence = 1.0  # placeholder
        return label, confidence

    def gpt_classifier(self, summary: str, abstract: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "developer",
                 "content": "Annotate whether a sentence or summary includes information not present in the original abstract. \n The sentence or summary contains external information that is not explicitly mentioned, paraphrased, or implied in the original abstract will be labeled as 'Yes'. \n The sentence or summary contains information that is explicitly stated or closely paraphrased from the original abstract will be labeled as 'No'."},
                {"role": "user",
                 "content": "Sentence or summary: " + summary + "\n Original abstract: " + abstract},
            ],
            max_tokens=self.max_tokens,
            temperature=0,
        )
        if response.choices[0].message.content == '':
            print('empty')
            result = 'Yes'
        else:
            result = response.choices[0].message.content
            result = re.sub(r'[^a-zA-Z0-9\s]', '', result)

        return result.lower().strip()

    def llama_classifier(self, summary: str, abstract: str) -> str:
        messages = [
            {"role": "system",
             "content": "Annotate whether a summary includes information not present in the original abstract. \n The summary contains external information that is not explicitly mentioned, paraphrased, or implied in the original abstract will be labeled as 'Yes'. \n The summary contains information that is explicitly stated or closely paraphrased from the original abstract will be labeled as 'No'."},
            {"role": "user",
             "content": "Summary: " + summary + "\n Original abstract: " + abstract + "\n Your annotation: "}
        ]

        outputs = self.generator(messages, do_sample=True, temperature=0.01, max_new_tokens=self.max_tokens)
        generated_text = outputs[0]['generated_text'][-1]['content']

        pattern = r'annotation is:\s*(Yes|No)\b'
        match = re.search(pattern, generated_text, flags=re.IGNORECASE)
        if match:
            final_answer = match.group(1)
        else:
            match = re.search(r'(Yes|No)\s*$', generated_text, flags=re.IGNORECASE)
            if match:
                final_answer = match.group(1)
            else:
                print("empty")
                final_answer = 'Yes'

        return final_answer.lower().strip()
