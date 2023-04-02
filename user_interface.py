# модуль user_interface.py
from load_models import load_main_model
from text_processing import clean_text, validate_input
from translation import TranslationService
from generate_response import generate_response_with_pipeline, ensemble_predictions
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
            response = generate_response_with_pipeline(
                translated_prompt,
                model,
                tokenizer,
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

def user_interface(device, model, tokenizer, translation_service, back_translation_service, weights, num_beams, temperature):
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
                if not validate_input(user_input, min_length=1, max_length=512):
                    print("Ошибка: Ввод не соответствует заданным критериям.")
                    print("Пожалуйста, введите вопрос длиной от 1 до 512 символов.")
                    continue

                user_prompt = clean_text(user_input)

                answer = process_user_input(
                    device,
                    settings,
                    user_prompt,
                    model,
                    tokenizer,
                    translation_service,
                    back_translation_service,
                    weights,
                    max_length=settings["max_length"],
                    temperature=settings["temperature"],
                    num_beams=num_beams  # Замените settings["top_k"] на num_beams
                )
                print(Fore.GREEN + "Вопрос пользователя: " + Fore.RESET + user_input)
                print(Fore.BLUE + "Ответ: " + Fore.RESET + answer)

        elif user_choice == "2":
            print("Текущие настройки:")
            for key, value in settings.items():
                print(f"{key}: {value}")

            print("Выберите настройку для изменения:")
            print("1. Максимальная длина ответа")
            print("2. Температура")
            print("3. Top-k")
            print("4. Назад")

            setting_choice = input("Введите номер настройки: ")

            if setting_choice == "1":
                new_value = int(input("Введите новое значение максимальной длины ответа (10-300): "))
                if 10 <= new_value <= 300:
                    settings["max_length"] = new_value
                else:
                    print("Некорректное значение, оставляем значение по умолчанию.")

            elif setting_choice == "2":
                new_value = float(input("Введите новое значение температуры (0.1-2.0): "))
                if 0.1 <= new_value <= 2.0:
                    settings["temperature"] = new_value
                else:
                    print("Некорректное значение, оставляем значение по умолчанию.")

            elif setting_choice == "3":
                new_value = int(input("Введите новое значение Top-k (0-50): "))
                if 0 <= new_value <= 50:
                    settings["top_k"] = new_value
                else:
                    print("Некорректное значение, оставляем значение по умолчанию.")

            elif setting_choice == "4":
                pass

            else:
                print("Некорректный выбор, попробуйте еще раз.")

        elif user_choice == "3":
            print("Выход...")
            break

        else:
             print("Некорректный выбор, попробуйте еще раз.")

    print("Спасибо за использование нашей системы. До свидания!")
