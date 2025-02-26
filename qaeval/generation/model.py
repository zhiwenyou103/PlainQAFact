import math
import torch
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

class QuestionGenerationModel:
    def __init__(self,
                 model_path: str,
                 cuda_device: int = 0,
                 batch_size: int = 8,
                 silent: bool = True):
        """
        Initialize the question generation model
        
        Args:
            model_path (str): Path to the fine-tuned model
            cuda_device (int): GPU device number
            batch_size (int): Batch size for generation
            silent (bool): Whether to show progress bar
        """
        self.device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
        
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = BartTokenizer.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()
        
        self.batch_size = batch_size
        self.silent = silent

    def generate(self, text: str, start: int, end: int) -> str:
        """
        Generate a single question
        
        Args:
            text (str): Full context text
            start (int): Start index of answer
            end (int): End index of answer
        
        Returns:
            Generated question
        """
        return self.generate_all([(text, start, end)])[0]

    def generate_all(self, inputs: List[Tuple[str, int, int]]) -> List[str]:
        """
        Generate questions for multiple inputs
        
        Args:
            inputs (List[Tuple]): List of (text, start_index, end_index) tuples
        
        Returns:
            List of generated questions
        """
        outputs = []
        num_batches = int(math.ceil(len(inputs) / self.batch_size))
        generator = range(0, len(inputs), self.batch_size)
        if not self.silent:
            generator = tqdm(generator, total=num_batches, desc='Generating questions')

        for i in generator:
            batch = inputs[i:i + self.batch_size]
            batch_contexts = []
            batch_answers = []
            
            for text, start, end in batch:
                context = text
                answer = text[start:end]
                batch_contexts.append(context)
                batch_answers.append(answer)

            encodings = self.tokenizer(
                batch_contexts, 
                batch_answers, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            )

            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=64,
                    num_return_sequences=1,
                    do_sample=False
                )

            batch_questions = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            outputs.extend(batch_questions)
        
        return outputs