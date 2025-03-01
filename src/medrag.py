import os
import re
import json
import torch
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
sys.path.append("src")
from src.utils import RetrievalSystem, DocExtracter
from template import *

class MedRAG:

    def __init__(self, cuda_device=0, retrieval=True, follow_up=False, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None, corpus_cache=False, HNSW=False):
        self.retrieval = retrieval
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None
        self.device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
        if retrieval:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW, device=self.device)
        else:
            self.retrieval_system = None
        self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                    "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}
        
        self.follow_up = follow_up
        self.answer = self.medrag_answer

    def medrag_answer(self, question, options=None, k=32, rrf_k=100, save_dir = None, snippets=None, snippets_ids=None, **kwargs):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        '''

        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''

        # retrieve relevant snippets
        if self.retrieval:
            if snippets is not None:
                retrieved_snippets = snippets[:k]
                scores = []
            elif snippets_ids is not None:
                if self.docExt is None:
                    self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
                retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                scores = []
            else:
                assert self.retrieval_system is not None
                retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
        else:
            retrieved_snippets = []
            scores = []
            contexts = []

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        return retrieved_snippets, scores

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)
