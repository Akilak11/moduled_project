# модуль translation
#содержит класс TranslationService, который предоставляет функциональность для перевода текста, возможно с использованием классов из модуля translation_models.py. Этот модуль, вероятно, был создан для удобной интеграции функциональности машинного перевода в вашем проекте.

import torch
import transformers
import config


class ForwardTranslationService:
    def __init__(self, model_name=config.TRANSLATION_MODEL_NAME, device=config.DEVICE):
        self.model_name = model_name
        self.device = device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def translate(self, text, max_length):
            try:
                # Токенизация текста
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

                # Перевод текста
                with torch.no_grad():
                    input_ids = inputs["input_ids"].to(self.device)
                    attention_mask = inputs["attention_mask"].to(self.device)
                    output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
                    translated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    
                return translated_text
            except:
                print("Ошибка при переводе текста.")
                return None

    def __call__(self, text, max_length=512):
        return self.translate(text, max_length)


class BackTranslationService:
    def __init__(self, model_name=config.BACK_TRANSLATION_MODEL_NAME, device=config.DEVICE):
        self.model_name = model_name
        self.device = device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def back_translate(self, text, forward_translation_service, max_length):
        # В этой функции используйте экземпляр forward_translation_service для выполнения прямого перевода
        try:
            translated_text = forward_translation_service.translate(text, max_length)

            inputs = self.tokenizer(translated_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            with torch.no_grad():
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
                back_translated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            return back_translated_text
        except:
            print("Ошибка при обратном переводе текста.")
            return None

    def __call__(self, text, forward_translation_service, max_length=512):
        return self.back_translate(text, forward_translation_service, max_length)
