# модуль user_interface.py
from load_models import load_models, load_translation_models
from text_processing import clean_text, validate_input
from translation import TranslationService
from generate_response import generate_response, ensemble_predictions
from colorama import Fore

import cpuinfo

# Получить информацию о процессоре
info = cpuinfo.get_cpu_info()

# Получить информацию о поддержке инструкций процессора
sse2_supported = 'sse2' in info.get('flags', '')
avx_supported = 'avx' in info.get('flags', '')
avx2_supported = 'avx2' in info.get('flags', '')
fma_supported = 'fma' in info.get('flags', '')

print(info)

def process_user_input(
    device,
    settings,
    user_prompt,
    model,
    tokenizer,
    translation_service,
    back_translation_service,
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
            translated_prompt = translation_service.translate(user_prompt, max_length=512)

            generated_responses = []
            response = generate_response(
                translated_prompt,
                [model],
                ensemble=False,
                back_translate=True,
                num_beams=num_beams,
                temperature=temperature
            )
            generated_responses.append(response)

            ensemble_response = ensemble_predictions(generated_responses, weights)

            translated_response = back_translation_service.translate(ensemble_response, max_length=512)

            return translated_response

        except Exception as e:
            print("Ошибка во время генерации ответа. Повторяю...")
            retries += 1

    print("Не удалось сгенерировать ответ. Попробуйте еще раз.")
    return ""

def user_interface(device, main_model, main_tokenizer, main_model_name, translation_model, back_translation_model, text_generator, weights, num_beams, temperature):
    print("Добро пожаловать в систему генерации ответов на основе искусственного интеллекта!")

    # Настройки по умолчанию
    settings = {
        "max_length": 50,
        "temperature": 1.0,
        "top_k": 0,
    }

    while True:
        print("Выберите опцию:")
        print("1. Задать вопрос")
        print("2. Изменить настройки")
        print("3. Выйти")

        user_choice = input("Введите номер опции: ")

        if user_choice == "1":
            user_input = input("Введите ваш вопрос: ")
            if user_input.lower() == "exit":
                print("Спасибо за использование нашей системы. До свидания!")
                break
            else:
                if not validate_input(user_input):
                    print("Пожалуйста, введите корректный текст.")
                    continue

                response = process_user_input(
                    device,
                    settings,
                    user_input,
                    model,
                    tokenizer,
                    translation_service,
                    back_translation_service,
                    weights,
                    num_beams=num_beams,
                    temperature=temperature
                )
                print(f"Ответ: {response}")

        elif user_choice == "2":
            print("Изменить настройки")
            # Здесь можно добавить функционал для изменения настроек
            print("Этот функционал пока не доступен")

        elif user_choice == "3":
            print("Спасибо за использование нашей системы. До свидания!")
            break

        else:
            print("Неверный ввод. Пожалуйста, введите номер одной из предложенных опций.")
