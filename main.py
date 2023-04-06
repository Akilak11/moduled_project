# модуль main
import torch
import os
import platform
import sys
import cpuinfo
import config
from load_models import load_models, load_translation_models
from translation import TranslationService
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
models, tokenizers = load_models(config.PARAMETERS)
translation_model, back_translation_model = load_translation_models()

# Создание экземпляров TranslationService
translation_service = TranslationService(PARAMETERS['translation_model_name'], PARAMETERS['device'])

# Настройки для генерации ответа
settings = {
    'ensemble': False,  # Использовать ансамбль или нет
    'back_translate': False,  # Использовать обратный перевод или нет
    'weights': None  # Веса для ансамбля (если используется ансамбль)
}

if __name__ == "__main__":
    user_interface(
        model,
        tokenizer,
        translation_service,
        translation_model,
        back_translation_model,
        settings['weights'],
        settings
    )
    
#clear_gpu_cache()
