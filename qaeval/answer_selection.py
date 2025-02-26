import spacy
import openai
from collections import namedtuple
from spacy.tokens import Span
from typing import List
import configparser
import re
import os
from transformers import pipeline
# config = configparser.ConfigParser()
# config.read('config.ini')

NP_CHUNKS_STRATEGY = 'np-chunks'
MAX_NP_STRATEGY = 'max-np'
NER_STRATEGY = 'ner'
ALL_STRATEGY = 'all'
LLM_KEYWORDS_STRATEGY = 'llm-keywords'
GPT_KEYWORDS_STRATEGY = 'gpt-keywords'

STRATEGIES = [NP_CHUNKS_STRATEGY, MAX_NP_STRATEGY, NER_STRATEGY, ALL_STRATEGY, LLM_KEYWORDS_STRATEGY, GPT_KEYWORDS_STRATEGY]
AnswerOffsets = namedtuple('Answer', ['start', 'end', 'sent_start', 'sent_end', 'text'])

class AnswerSelector(object):
    def __init__(self, strategy: str, generator=None):
        if strategy not in STRATEGIES:
            raise Exception(f'Unknown strategy: {strategy}')
        self.strategy = strategy
        self.nlp = spacy.load('en_core_web_sm')
        if self.strategy == LLM_KEYWORDS_STRATEGY:
            self.generator = generator
        elif self.strategy == GPT_KEYWORDS_STRATEGY:
            self.client = generator

    def _get_np_chunks_answers(self, sentence: Span) -> List[AnswerOffsets]:
        chunks = []
        for chunk in sentence.noun_chunks:
            chunks.append(AnswerOffsets(
                chunk.start_char,
                chunk.end_char,
                sentence.start_char,
                sentence.end_char,
                str(chunk)
            ))
        return chunks

    def _get_max_np_answers(self, sentence: Span) -> List[AnswerOffsets]:
        root = sentence.root
        nodes = [root]
        nps = []

        while nodes:
            node = nodes.pop()
            recurse = True
            if node.pos_ in ['NOUN', 'PROPN']:
                min_index = node.i
                max_index = node.i
                stack = [node]
                while stack:
                    current = stack.pop()
                    min_index = min(min_index, current.i)
                    max_index = max(max_index, current.i)
                    for child in current.children:
                        stack.append(child)

                sent_start_index = sentence[0].i
                num_tokens = max_index - min_index + 1
                if num_tokens <= 7:
                    recurse = False
                    span = sentence[min_index - sent_start_index:max_index + 1 - sent_start_index]
                    nps.append(AnswerOffsets(
                        span.start_char,
                        span.end_char,
                        sentence.start_char,
                        sentence.end_char,
                        str(span)
                    ))

            if recurse:
                for child in node.children:
                    nodes.append(child)

        nps.sort(key=lambda offsets: offsets.start)
        return nps

    def _get_ner_answers(self, sentence: Span) -> List[AnswerOffsets]:
        ners = []
        for entity in sentence.ents:
            if entity.label_ in ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART']:
                ners.append(AnswerOffsets(
                    entity.start_char,
                    entity.end_char,
                    sentence.start_char,
                    sentence.end_char,
                    str(entity)
                ))
        return ners

    def _get_all_answers(self, sentence: Span) -> List[AnswerOffsets]:
        answers = set()
        answers |= set(self._get_np_chunks_answers(sentence))
        answers |= set(self._get_max_np_answers(sentence))
        answers |= set(self._get_ner_answers(sentence))
        answers = sorted(answers, key=lambda answer: (answer.start, answer.end))
        return answers

    def _call_gpt(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "developer", "content": "QA-based metrics compare information units between the summary and source, so it is thus necessary to first extract such units, or answers, from the given summary. Please extract answers or information units from a plain language summary."},
                {"role": "user", "content": "Extract a comma-separated list of the most important keywords from the following text: \n" + prompt},
            ],
            max_tokens=512,
            temperature=0,
        )
        return response.choices[0].message.content
    
    def _call_llama(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "QA-based metrics compare information units between the summary and source, so it is thus necessary to first extract such units, or answers, from the given summary. Please extract answers or information units from a plain language summary."},
            {"role": "user", "content": "Extract a comma-separated list of the most important keywords from the following text: \n" + prompt},
        ]
        outputs = self.generator(messages, do_sample=True, temperature=0.01, max_new_tokens=512)
        generated_text = outputs[0]['generated_text'][-1]['content']
        parts = generated_text.split(":\n\n", 1)
        match = re.search(r"(\b[\w\s\-]+(?:,\s*[\w\s\-]+)+)", generated_text)
        if len(parts) == 2:
            generated_text = parts[1] 
        elif match:
            generated_text = match.group(1)
        else:
            generated_text = generated_text
        return generated_text

    def _get_llmam_keywords_answers(self, text: str) -> List[AnswerOffsets]:
        """
        Prompts the LLM to extract a comma-separated list of keywords from the text.
        Then, for each keyword, finds its first occurrence in the text and determines the sentence boundaries.
        """
        keywords_str = self._call_llama(text)
        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
        print(f"Keywords extracted by the LLM: {keywords}")
        answers = []
        for keyword in keywords:
            start = text.find(keyword)
            if start != -1:
                end = start + len(keyword)
                doc = self.nlp(text)
                sent_start = 0
                sent_end = len(text)
                for sent in doc.sents:
                    if start >= sent.start_char and end <= sent.end_char:
                        sent_start = sent.start_char
                        sent_end = sent.end_char
                        break
                answers.append(AnswerOffsets(start, end, sent_start, sent_end, keyword))
        return answers
    
    def _get_gpt_keywords_answers(self, text: str) -> List[AnswerOffsets]:
        """
        Prompts the LLM to extract a comma-separated list of keywords from the text.
        Then, for each keyword, finds its first occurrence in the text and determines the sentence boundaries.
        """
        keywords_str = self._call_gpt(text)
        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
        print(f"Keywords extracted by the LLM: {keywords}")
        answers = []
        for keyword in keywords:
            start = text.find(keyword)
            if start != -1:
                end = start + len(keyword)
                doc = self.nlp(text)
                sent_start = 0
                sent_end = len(text)
                for sent in doc.sents:
                    if start >= sent.start_char and end <= sent.end_char:
                        sent_start = sent.start_char
                        sent_end = sent.end_char
                        break
                answers.append(AnswerOffsets(start, end, sent_start, sent_end, keyword))
        return answers

    def select(self, text: str) -> List[AnswerOffsets]:
        """
        Depending on the strategy, returns a list of AnswerOffsets (i.e. extracted keywords or phrases)
        from the input text.
        """
        if self.strategy == LLM_KEYWORDS_STRATEGY:
            return self._get_llmam_keywords_answers(text)
        elif self.strategy == GPT_KEYWORDS_STRATEGY:
            return self._get_gpt_keywords_answers(text)
        else:
            doc = self.nlp(text)
            answers = []
            for sent in doc.sents:
                if self.strategy == NP_CHUNKS_STRATEGY:
                    answers.extend(self._get_np_chunks_answers(sent))
                elif self.strategy == MAX_NP_STRATEGY:
                    answers.extend(self._get_max_np_answers(sent))
                elif self.strategy == NER_STRATEGY:
                    answers.extend(self._get_ner_answers(sent))
                elif self.strategy == ALL_STRATEGY:
                    answers.extend(self._get_all_answers(sent))
                else:
                    raise Exception(f'Unknown strategy: {self.strategy}')
            return answers

    def select_all(self, text_list: List[str]) -> List[List[AnswerOffsets]]:
        return [self.select(text) for text in text_list]
