# модуль user_interface
import platform
import os
import sys
import cpuinfo
from config import PARAMETERS
from load_models import load_models, load_translation_models
from text_processing import clean_text, validate_input
from translation import TranslationService
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
    model, tokenizer = load_models(PARAMETERS)
    translation_model, back_translation_model = load_translation_models(PARAMETERS)

    # Создание экземпляров TranslationService
    translation_service = TranslationService(PARAMETERS['translation_model_name'], PARAMETERS['device'])

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
                model,
                tokenizer,
                translation_service,
                translation_model,
                back_translation_model
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

'''def process_user_input(user_input, settings, model, tokenizer, translation_service, translation_model, back_translation_model, weights):
    cleaned_input = clean_text(user_input)
    if not validate_input(cleaned_input):
        return "Ошибка: Введенный текст не соответствует требованиям."

    translated_input = translation_service.translate_text(cleaned_input, translation_model)
    generated_response = generate_response(translated_input, model, tokenizer, weights)

    back_translated_response = translation_service.translate_text(generated_response, back_translation_model, reverse=True)
    return back_translated_response
'''
