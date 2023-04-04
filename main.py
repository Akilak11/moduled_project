# модуль main.py
import os
import sys
import torch
import platform
from colorama import init, Fore

from load_models import load_main_model
from resource_manager import enable_memory_growth
from translation import TranslationService
from user_interface import user_interface
from utils import check_model_files, load_main_model, download_model
from config import MAIN_MODEL, DEVICE

init()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Работает на GPU:", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("Работает на CPU")

translation_service = TranslationService("Helsinki-NLP/opus-mt-en-ru", "cuda")
back_translation_service = TranslationService("Helsinki-NLP/opus-mt-ru-en", "cuda")

weights = [1.0, 0.8] 
num_beams = 5
temperature = 1.0

print("system_info: CPU = {}, RAM = {} GB, OS = {}, Python = {}".format(
    platform.processor(),
    round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 2),
    platform.system(),
    sys.version
))

print("Загрузка моделей...")

MAIN_MODEL = main_model
main_model, main_tokenizer = load_main_model()
models.append(main_model)
tokenizers.append(main_tokenizer)

print("Модели успешно загружены.")

if __name__ == "__main__":
    enable_memory_growth()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    main_model, main_tokenizer = load_main_model()
    translation_service = TranslationService("Helsinki-NLP/opus-mt-en-ru", "cuda")
    back_translation_service = TranslationService("Helsinki-NLP/opus-mt-ru-en", "cuda")
    num_beams = 5
    temperature = 1.0
    user_interface(device, main_model, main_tokenizer, translation_service, back_translation_service, weights)
