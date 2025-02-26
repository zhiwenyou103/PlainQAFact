from typing import Dict, List, Set
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from qaeval.scoring.scorers import Scorer


class LERCModel(nn.Module):
    def __init__(self, pretrained_model_name='bert-large-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.score_head = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        score = self.score_head(pooled_output)
        return score.squeeze(-1)


class LERCScorer(Scorer):
    def __init__(self, model_path: str, pretrained_path: str, cuda_device: int, batch_size: int = 8) -> None:
        self.device = torch.device(f"cuda:{cuda_device}" if cuda_device >= 0 else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased') # 'bert-base-uncased'
        self.batch_size = batch_size
        
    def keys(self) -> Set[str]:
        return {'lerc'}
    
    def _prepare_input(self, context: str, question: str, reference: str, candidate: str) -> Dict:
        # Format input following LERC paper
        text = f"Context: {context} Question: {question} Reference: {reference} Candidate: {candidate}"
        encoded = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return encoded
    
    def _batch_predict(self, batch_inputs: List[Dict]) -> List[float]:
        with torch.no_grad():
            input_ids = torch.cat([x['input_ids'] for x in batch_inputs]).to(self.device)
            attention_mask = torch.cat([x['attention_mask'] for x in batch_inputs]).to(self.device)
            scores = self.model(input_ids, attention_mask)
            return scores.cpu().tolist()

    def _score_single_ref(
        self,
        context: str,
        questions: List[str],
        answers: List[str],
        predictions: List[str],
        probabilities: List[float],
        null_probabilities: List[float]
    ) -> List[Dict[str, float]]:
        input_dicts = []
        indices = []

        for i, (answer, question, prediction, probability, null_probability) in enumerate(
            zip(answers, questions, predictions, probabilities, null_probabilities)
        ):
            encoded_input = self._prepare_input(context, question, answer, prediction)
            input_dicts.append(encoded_input)
            indices.append(i)

        all_scores = []
        for i in range(0, len(input_dicts), self.batch_size):
            batch = input_dicts[i:i + self.batch_size]
            batch_scores = self._batch_predict(batch)
            all_scores.extend(batch_scores)

        scores = [0.0] * len(questions)
        for i, score in zip(indices, all_scores):
            scores[i] = score
        
        return [{'lerc': s} for s in scores]