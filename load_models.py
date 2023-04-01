import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_main_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Устанавливаем pad_token равным eos_token
    return model, tokenizer

def load_translation_models():
    translation_model_name = "Helsinki-NLP/opus-mt-ru-en"
    back_translation_model_name = "Helsinki-NLP/opus-mt-en-ru"

    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name, src_lang="ru", tgt_lang="en")
    translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)
    translation_tokenizer.pad_token = translation_tokenizer.eos_token

    back_translation_tokenizer = MarianTokenizer.from_pretrained(back_translation_model_name, src_lang="en", tgt_lang="ru")
    back_translation_model = MarianMTModel.from_pretrained(back_translation_model_name).to(device)
    back_translation_tokenizer.pad_token = back_translation_tokenizer.eos_token

    return translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer
