#модуль user_interface.py

from main import (
    models,
    tokenizers,
    translation_model,
    translation_tokenizer,
    back_translation_model,
    back_translation_tokenizer,
    weights,
    process_user_input
)

from translation import load_translation_models, translate_text
from code_processing import separate_code_and_explanations, combine_code_and_translated_explanations
from text_processing import clean_text

import cpuinfo

# Получить информацию о процессоре
info = cpuinfo.get_cpu_info()

# Получить информацию о поддержке инструкций процессора
sse2_supported = 'sse2' in info.get('flags', '')
avx_supported = 'avx' in info.get('flags', '')
avx2_supported = 'avx2' in info.get('flags', '')
fma_supported = 'fma' in info.get('flags', '')

print(info)
#Функция user_interface содержит основной цикл программы, который выполняется, пока пользователь не выберет опцию выхода.
#Внутри цикла пользователь может выбрать опцию для задания вопроса, изменения настроек или выхода из программы.
#Если пользователь выбирает опцию задания вопроса, то его вводится входной текст, который очищается и отправляется на обработку моделями, а затем выводится сгенерированный ответ.
def user_interface(device):
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
                user_prompt = separate_code_and_explanations(user_prompt)
                user_prompt = combine_code_and_translated_explanations(user_prompt)

                answer = process_user_input(
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
                    max_length=settings["max_length"],
                    temperature=settings["temperature"],
                    num_beams=settings["top_k"]
                )
                print("Ответ:")
                print(answer)

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
