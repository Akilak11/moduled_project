#модуль text_generator

#text_generator.py: Этот модуль создан для генерации текста на основе заданного ввода. Класс TextGenerator использует предобученные модели кодировщика и декодировщика для преобразования входного текста в векторное представление и последующей генерации текста на основе этого векторного представления. Возможно, этот модуль был создан для реализации генерации текста с использованием моделей Transformer, таких как GPT или BERT.


import transformers
import torch
from typing import List, Union
#from config import DEVICE, ENCODER_MODEL_NAME, DECODER_MODEL_NAME, GENERATION_MAX_LENGTH, GENERATION_MIN_LENGTH, GENERATION_TEMPERATURE

class TextGenerator:
    def __init__(self, encoder_model_name: str, decoder_model_name: str, device: str, generation_max_length: int, generation_temperature: float):
        self.encoder_tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_model_name)
        self.decoder_tokenizer = transformers.AutoTokenizer.from_pretrained(decoder_model_name)
        self.encoder_model = transformers.AutoModel.from_pretrained(encoder_model_name).to(device)
        self.decoder_model = transformers.AutoModelWithLMHead.from_pretrained(decoder_model_name).to(device)
        self.encoder_model.eval()
        self.decoder_model.eval()
        self.device = device
        self.generation_max_length = generation_max_length
        #self.generation_min_length = generation_min_length
        self.generation_temperature = generation_temperature

    def generate_response(self, input_text: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(input_text, str):
            input_text = [input_text]

        # Токенизация текста и получение векторных представлений
        input_ids = self.encoder_tokenizer.batch_encode_plus(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt")["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.encoder_model(input_ids)
            vector_repr = outputs[0][:, 0, :]

        # Генерация текста на основе векторного представления
        generated_sequences = self.decoder_model.generate(
            input_ids=None,
            max_length=self.generation_max_length,
            min_length=self.generation_min_length,
            num_return_sequences=1,
            temperature=self.generation_temperature,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
            bos_token_id=self.decoder_tokenizer.bos_token_id,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            decoder_start_token_id=self.decoder_tokenizer.bos_token_id,
            encoder_hidden_states=vector_repr.unsqueeze(0),
            attention_mask=None,
            use_cache=None
        )

        generated_responses = []
        for sequence in generated_sequences:
            sequence = sequence.tolist()
            text = self.decoder_tokenizer.decode(sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            generated_responses.append(text)

        if len(generated_responses) == 1:
            return generated_responses[0]
        return generated_responses

    def __call__(self, input_text: Union[str, List[str]]) -> Union[str, List[str]]:
        return self.generate_response(input_text)
