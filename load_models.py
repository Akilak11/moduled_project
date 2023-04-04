#модуль load_models.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MarianTokenizer, MarianMTModel, GPT2Tokenizer, GPTNeoForCausalLM, GPT2LMHeadModel
from config import MAIN_MODEL, MODEL_PATHS
from utils import check_model_files, download_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(model_names):
    models = []
    tokenizers = []

    for model_name in model_names:
        model_directory = MODEL_PATHS[model_name]

        if not check_model_files(model_name, model_directory):
            print(f"Проверка файлов модели {model_name} не пройдена. Начинаю скачивание модели.")
            download_model(model_name, model_directory)

        if model_name == "EleutherAI/gpt-neo-2.7B":
            model = GPTNeoForCausalLM.from_pretrained(model_directory).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        elif model_name == "gpt2":
            model = GPT2LMHeadModel.from_pretrained(model_directory).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_directory).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenizer.pad_token = tokenizer.eos_token # Устанавливаем pad_token равным eos_token

        models.append(model)
        tokenizers.append(tokenizer)

    return models, tokenizers

def load_translation_models():
    translation_model_name = "Helsinki-NLP/opus-mt-ru-en"
    back_translation_model_name = "Helsinki-NLP/opus-mt-en-ru"

    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)

    back_translation_tokenizer = MarianTokenizer.from_pretrained(back_translation_model_name)
    back_translation_model = MarianMTModel.from_pretrained(back_translation_model_name).to(device)

    return translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer
