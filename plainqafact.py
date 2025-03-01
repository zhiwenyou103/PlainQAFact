from typing import Dict, List, Union
from transformers.data.metrics.squad_metrics import compute_f1
from qaeval.metric import QAEval
import bert_score
from transformers import pipeline
from bertscore_scorer import BertScoreScorer

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
            *args,
            **kwargs):
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
        super().__init__(cuda_device=cuda_device, *args, **kwargs)
        
        bertscore_scorer = BertScoreScorer(cuda_device=cuda_device, batch_size=scoring_batch_size)
        self.scorer.scorers.append(bertscore_scorer)
        
        self.answer_selection_strategy = answer_selection_strategy
        

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
