# модуль main
import os
import sys
import torch
import platform
from colorama import init, Fore

from load_models import load_models, load_translation_models
from resource_manager import enable_memory_growth, clear_gpu_cache
from code_processing import separate_code_and_explanations, combine_code_and_translated_explanations
from text_processing import clean_text, extract_code_and_explanations, validate_input
from translation_models import TranslationModel, TRANSLATION_MODEL_NAME, BACK_TRANSLATION_MODEL_NAME
from text_generator import TextGenerator, ENCODER_MODEL_NAME, DECODER_MODEL_NAME
from user_interface import user_interface
from config import MODEL_NAMES, DEVICE

init()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Работает на GPU:", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("Работает на CPU")

enable_memory_growth()

models, tokenizers = load_models(MODEL_NAMES)
translation_model = TranslationModel(TRANSLATION_MODEL_NAME, device)
back_translation_model = TranslationModel(BACK_TRANSLATION_MODEL_NAME, device)
text_generator = TextGenerator(ENCODER_MODEL_NAME, DECODER_MODEL_NAME, DEVICE)

weights = [1.0, 0.8]
num_beams = 5
temperature = 1.0

print("system_info: CPU = {}, RAM = {} GB, OS = {}, Python = {}".format(
    platform.processor(),
    round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 2),
    platform.system(),
    sys.version
))

print("Модели успешно загружены.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    main_model, main_tokenizer = models[0], tokenizers[0]  # Используем первую модель и токенизатор (например, GPT-2)
    user_interface(device, main_model, main_tokenizer, translation_model, back_translation_model, text_generator, weights, num_beams, temperature)

clear_gpu_cache()
