#модуль main.py

import os
import sys
import torch
import platform
from colorama import init, Fore

from load_models import load_main_model, load_translation_models
from generate_response import generate_response_with_pipeline, ensemble_predictions
from utils import check_model_files
from code_processing import separate_code_and_explanations, combine_code_and_translated_explanations
from resource_manager import enable_memory_growth, get_tokenizer_and_model, clear_model, get_models_list
from text_processing import clean_text
from translation import translate_text
from user_interface import user_interface

init()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Работает на GPU:", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("Работает на CPU")

translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer = load_translation_models()

models, tokenizers = [], []
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

main_model, main_tokenizer = load_main_model()
models.append(main_model)
tokenizers.append(main_tokenizer)

print("Модели успешно загружены.")

print(Fore.GREEN + "Вопрос пользователя: " + Fore.RESET)

def process_user_input(
    device,
    settings,
    user_prompt,
    models,
    tokenizers,
    translation_model,
    translation_tokenizer,
    back_translation_model,
    back_translation_tokenizer,
    weights,
    user_language="ru",
    max_length=512,
    max_retries=3,
    num_beams=5,
    temperature=1.0
):
    if len(user_prompt) > max_length:
        print(f"Длина ввода слишком большая. Максимальная длина: {max_length}")
        return ""

    retries = 0
    while retries < max_retries:
        try:
            translated_prompt = translate_text(user_prompt, translation_model, translation_tokenizer, target_language="en")

            generated_responses = []
            for model, tokenizer in zip(models, tokenizers):
                response = generate_response_with_pipeline(
                    translated_prompt,
                    model,
                    tokenizer,
                    num_beams=num_beams,
                    temperature=temperature
                )
                generated_responses.append(response)

            ensemble_response = ensemble_predictions(generated_responses, weights)

            translated_response = translate_text(ensemble_response, back_translation_model, back_translation_tokenizer, target_language="ru")

            return translated_response

        except Exception as e:
            print("Ошибка во время генерации ответа. Повторяю...")
            retries += 1

    print("Не удалось сгенерировать ответ. Попробуйте еще раз.")
    return ""

while True:
    user_prompt = input("Введите ваш вопрос (или 'exit' для выхода): ")

    if user_prompt.lower() == "exit":
        break

    answer = process_user_input(
        user_prompt,
        models,
        tokenizers,
        translation_model,
        translation_tokenizer,
        back_translation_model,
        back_translation_tokenizer,
        weights
    )

    print(Fore.BLUE + "Ответ: " + Fore.RESET + answer)

if __name__ == "__main__":
    enable_memory_growth()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    user_interface(device)
    main()