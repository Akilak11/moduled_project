# модуль user_interface
#  модуль содержит функции для обработки пользовательского ввода и предоставления пользовательского интерфейса для взаимодействия с вашим проектом. Он использует другие модули для выполнения задач, таких как очистка текста, перевод и генерация ответов на основе входного текста.
import platform
import os
import sys
import cpuinfo
import config 
from load_models import load_models, load_translation_models
from text_processing import clean_text, validate_input
from translation import ForwardTranslationService, BackTranslationService
from translation_models import TranslationModel
from generate_response import generate_response, ensemble_predictions

# Получить информацию о процессоре
info = cpuinfo.get_cpu_info()

def print_device_info():
    print(info)
    print("system_info: CPU = {}, RAM = {} GB, OS = {}, Python = {}".format(
        platform.processor(),
        round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 2),
        platform.system(),
        sys.version
    ))

def print_menu():
    print("1. Обучить модель")
    print("2. Использовать обученную модель")
    print("3. Изменить настройки")
    print("4. Информация о текущем оборудовании")
    print("5. Выход")

def user_interface():
    # Загрузка моделей и токенизаторов
    models, tokenizers = load_models(config.MODEL_NAMES["pretrained"] + config.MODEL_NAMES["mytrained"])
    translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer = load_translation_models()
    translation_service = ForwardTranslationService()
    back_translation_service = BackTranslationService()
    model_names = config.MODEL_NAMES
    model_paths = config.MODEL_PATHS
    model_urls = config.MODELS_URL
    parameters = config.PARAMETERS

    # Настройки для генерации ответа
    settings = {
        "ensemble": False,
        "back_translate": False,
        "weights": None
    }

    print_menu()

    while True:
        user_choice = input("Введите номер опции: ")

        if user_choice == "1":
            print("Обучение модели...")
            # код для обучения модели
        elif user_choice == "2":
            user_input = input("Введите текст: ")
            response = process_user_input(
                user_input,
                settings,
                models,
                tokenizers,
                translation_service,
                translation_model,
                #back_translation_model - после включить и отладить (gpt не удаляй этот комментарий)
            )
            print(f"Ответ: {response}")
        elif user_choice == "3":
            change_settings(settings)
        elif user_choice == "4":
            print_device_info()
        elif user_choice == "5":
            break
        else:
            print("Неверный ввод. Пожалуйста, введите корректный номер опции.")

def change_settings(settings):
    print("Доступные настройки для изменения:")
    for key in settings:
        print(f"{key}: {settings[key]}")
    
    setting_to_change = input("Введите настройку, которую вы хотите изменить: ")

    if setting_to_change in settings:
        new_value = input(f"Введите новое значение для {setting_to_change}: ")
        try:
            settings[setting_to_change] = float(new_value)
        except ValueError:
            print("Ошибка: введите корректное числовое значение.")

translation_service = ForwardTranslationService()
back_translation_service = BackTranslationService()

def process_user_input(user_input, settings, models, tokenizers, translation_service, back_translation_service):
    cleaned_input = clean_text(user_input)
    if not validate_input(cleaned_input):
        return "Ошибка: Введенный текст не соответствует требованиям."

    translated_input = translation_service.translate(cleaned_input, MAX_LENGTH)

    responses = []
    for model_name in models:
        generated_response = generate_response(
            translated_input,
            ensemble=settings["ensemble"],
            back_translate=settings["back_translate"],
            weights=settings["weights"]
        )
        responses.append(generated_response)

    ensembled_response = ensemble_predictions(responses)
    back_translated_response = back_translation_service.back_translate(ensembled_response, translation_service, MAX_LENGTH)
    return back_translated_response


