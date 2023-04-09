# модуль main
import torch
import os
import platform
import sys
import cpuinfo
import config
#import transformers
from load_models import load_models, load_translation_models, load_pretrained_models
from translation import ForwardTranslationService, BackTranslationService
from translation_models import TranslationModel
from user_interface import user_interface, change_settings, process_user_input
from colorama import init, Fore
from resource_manager import enable_memory_growth, clear_gpu_cache

#init()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Работает на GPU:", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("Работает на CPU")

enable_memory_growth()

# Загрузка моделей и токенизаторов
models, tokenizers = load_pretrained_models()
translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer = load_translation_models()

# Создание экземпляров TranslationService
forward_translation_service = ForwardTranslationService(config.TRANSLATION_MODEL_NAME, device)
back_translation_service = BackTranslationService(config.TRANSLATION_MODEL_NAME, device)

# Настройки для генерации ответа
settings = {
    'ensemble': False,  # Использовать ансамбль или нет
    'back_translate': False,  # Использовать обратный перевод или нет
    'weights': None  # Веса для ансамбля (если используется ансамбль)
}

if __name__ == "__main__":
    user_interface()

#clear_gpu_cache()
