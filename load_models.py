#модуль load_models.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Функция load_main_model принимает имя основной модели и имя токенизатора, и загружает модель и токенизатор с помощью Hugging Face Transformers.
#Модель затем отправляется на устройство, доступное для выполнения (GPU или CPU).
def load_main_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Устанавливаем pad_token равным eos_token
    return model, tokenizer

def load_translation_models():
    translation_model_name = "Helsinki-NLP/opus-mt-ru-en"
    back_translation_model_name = "Helsinki-NLP/opus-mt-en-ru"

    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)

    back_translation_tokenizer = MarianTokenizer.from_pretrained(back_translation_model_name)
    back_translation_model = MarianMTModel.from_pretrained(back_translation_model_name).to(device)

    return translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer
