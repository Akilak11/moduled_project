# модуль load_models
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, MarianTokenizer, MarianMTModel, GPT2Tokenizer, GPTNeoForCausalLM, GPT2LMHeadModel
from config import MODELS_PATH, MODEL_NAMES, ENCODER_MODEL_NAME, DECODER_MODEL_NAME, TRANSLATION_MODEL_NAME, BACK_TRANSLATION_MODEL_NAME, DEVICE
from utils import setup_models, is_file_available, download_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(model_names_list):
    models = []
    tokenizers = []

    for model_name in model_names_list:
        for name, model_id in MODEL_NAMES.items():
            if name == model_name:
                model_path = MODELS_PATH / model_id

                # Check if the model files exist in the cache and download them if not
                download_models(model_path.parent, [model_name], MODELS_URL)

                if model_name == "EleutherAI/gpt-neo-2.7B":
                    model = GPTNeoForCausalLM.from_pretrained(model_path, cache_dir=model_path).to(DEVICE)
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                elif model_name == "gpt2":
                    model = GPT2LMHeadModel.from_pretrained(model_path, cache_dir=model_path).to(DEVICE)
                    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=model_path).to(DEVICE)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)

                tokenizer.pad_token = tokenizer.eos_token  # Устанавливаем pad_token равным eos_token

                models.append(model)
                tokenizers.append(tokenizer)

    return models, tokenizers

def load_translation_models():
    translation_model_name = TRANSLATION_MODEL_NAME
    back_translation_model_name = BACK_TRANSLATION_MODEL_NAME

    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name).to(DEVICE)

    back_translation_tokenizer = MarianTokenizer.from_pretrained(back_translation_model_name)
    back_translation_model = MarianMTModel.from_pretrained(back_translation_model_name).to(DEVICE)

    return translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer
