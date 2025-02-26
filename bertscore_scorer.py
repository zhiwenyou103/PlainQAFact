from typing import Dict, List, Set
import bert_score
import numpy as np
from qaeval.scoring.scorers.scorer import Scorer

class BertScoreScorer(Scorer):
    def __init__(self, cuda_device: int, batch_size: int = 8) -> None:
        self.device = f'cuda:{cuda_device}' if cuda_device >= 0 else 'cpu'
        self.batch_size = batch_size

    def keys(self) -> Set[str]:
        return {'bertscore'}

    def _score_single_ref(
        self,
        context: str,
        questions: List[str],
        answers: List[str],
        predictions: List[str],
        probabilities: List[float],
        null_probabilities: List[float]
    ) -> List[Dict[str, float]]:
        indices = []
        valid_answers = []
        valid_predictions = []
        
        for i, (answer, prediction, probability, null_probability) in \
                enumerate(zip(answers, predictions, probabilities, null_probabilities)):
            if probability > null_probability:
                valid_answers.append(answer)
                valid_predictions.append(prediction)
                indices.append(i)

        scores = [0.0] * len(questions)
        if valid_predictions:
            _, _, F1 = bert_score.score(valid_predictions, valid_answers, 
                                      lang="en", device=self.device)
            for idx, score in zip(indices, F1.tolist()):
                scores[idx] = score

        return [{'bertscore': s} for s in scores]