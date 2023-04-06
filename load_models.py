import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MarianTokenizer, MarianMTModel, GPT2Tokenizer, GPTNeoForCausalLM, GPT2LMHeadModel
from config import MODELS_PATH, ENCODER_MODEL_NAME, DECODER_MODEL_NAME, TRANSLATION_MODEL_NAME, BACK_TRANSLATION_MODEL_NAME
from utils import setup_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(model_names):
    setup_models()

    models = []
    tokenizers = []

    for model_name in model_names:
        model_path = MODELS_PATH / model_name

        if model_name == "EleutherAI/gpt-neo-2.7B":
            model = GPTNeoForCausalLM.from_pretrained(model_path).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        elif model_name == "gpt2":
            model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        tokenizer.pad_token = tokenizer.eos_token # Устанавливаем pad_token равным eos_token

        models.append(model)
        tokenizers.append(tokenizer)

    return models, tokenizers

def load_translation_models():
    translation_model_name = TRANSLATION_MODEL_NAME
    back_translation_model_name = BACK_TRANSLATION_MODEL_NAME

    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)

    back_translation_tokenizer = MarianTokenizer.from_pretrained(back_translation_model_name)
    back_translation_model = MarianMTModel.from_pretrained(back_translation_model_name).to(device)

    return translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer
