#модуль translation_models

import torch
import transformers
from config import DEVICE, BACK_TRANSLATION_MODEL_NAME, TRANSLATION_MODEL_NAME
from typing import List, Tuple

TRANSLATION_MAX_LENGTH = 512

class TranslationModel:
    def __init__(self, model_name: str = TRANSLATION_MODEL_NAME, device: torch.device = DEVICE):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def translate(self, text: str, max_length: int = TRANSLATION_MAX_LENGTH) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded

    def batch_translate(self, texts: List[str], batch_size: int = 32) -> List[str]:
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        results = []
        for batch in batches:
            input_ids = self.tokenizer.batch_encode_plus(batch, return_tensors="pt").to(self.device)
            outputs = self.model.generate(input_ids["input_ids"])
            decoded = [self.tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(len(batch))]
            results += decoded
        return results

    def back_translate(self, text: str) -> str:
        # Перевод текста на английский язык
        english_text = self.translate(text, TRANSLATION_MAX_LENGTH)

        # Обратный перевод английского текста на исходный язык
        back_translated_text = self.translate(english_text, TRANSLATION_MAX_LENGTH)

        return back_translated_text

    def close(self):
        self.model.cpu()
        torch.cuda.empty_cache()

#...

# Ваши другие функции и код для ансамблей моделей, если они нужны.


