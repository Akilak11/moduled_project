# модуль load_models
import torch
import os
from pathlib import Path
import transformers
#from transformers import AutoTokenizer, AutoModelForCausalLM, MarianTokenizer, MarianMTModel, GPT2Tokenizer, GPTNeoForCausalLM
from config import MODELS_PATH, PRETRAINED_MODEL_PATHS, MYTRAINED_MODEL_PATHS, MODEL_NAMES, ENCODER_MODEL_NAME, DECODER_MODEL_NAME, TRANSLATION_MODEL_NAME, BACK_TRANSLATION_MODEL_NAME, DEVICE, MODELS_URL
from utils import is_file_available, download_models, is_model_available_locally
#from user_interface import model_paths

from transformers import AutoTokenizer, AutoModelForCausalLM, MarianTokenizer, MarianMTModel, GPT2Tokenizer, GPTNeoForCausalLM, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name, model_paths):
    model_path = model_paths[model_name]

    if not is_model_available_locally(model_path):
        print(f"Model {model_name} not found in local storage. Downloading from Hugging Face Hub...")
        download_models(model_path.parent, [model_name], MODELS_URL)

    elif model_name == "EleutherAI/gpt-neo-2.7B":
        model = GPTNeoForCausalLM.from_pretrained(model_path).to(DEVICE)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.pad_token = tokenizer.eos_token  # Устанавливаем pad_token равным eos_token

    return model, tokenizer

def load_models(model_name_list, model_paths):
    models, tokenizers = [], []
    for model_name in model_name_list:
        model, tokenizer = load_model(model_name, model_paths)
        models.append(model)
        tokenizers.append(tokenizer)

    print("Модели загруженные в load_models: ", models)
    print("Токенизеры загруженные в load_models: ", tokenizers)

    assert len(models) > 0, "Список моделей пуст"
    assert len(tokenizers) > 0, "Список токенизеров пуст"
    return models, tokenizers

def load_pretrained_models():
    pretrained_model_names = [model_name for model_name_dict in MODEL_NAMES["pretrained"] for key, model_name_list in (model_name_dict.items() if isinstance(model_name_dict, dict) else [(None, [model_name_dict])]) for model_name in model_name_list]
    models, tokenizers = load_models(pretrained_model_names, PRETRAINED_MODEL_PATHS)
    return models, tokenizers

def load_mytrained_models():
    mytrained_model_names = MODEL_NAMES["mytrained"]
    models, tokenizers = load_models(mytrained_model_names, MYTRAINED_MODEL_PATHS)
    return models, tokenizers

def load_translation_models():
    translation_model_name = TRANSLATION_MODEL_NAME
    back_translation_model_name = BACK_TRANSLATION_MODEL_NAME

    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name).to(DEVICE)

    back_translation_tokenizer = MarianTokenizer.from_pretrained(back_translation_model_name)
    back_translation_model = MarianMTModel.from_pretrained(back_translation_model_name).to(DEVICE)

    return translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer
