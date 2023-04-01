# Импорт стандартных библиотек
import os
import sys

# Импорт сторонних библиотек
import torch
import platform
from colorama import init, Fore

from load_models import load_main_model, load_translation_models
# Импорт локальных модулей
from generate_response import generate_response_with_pipeline, generate_response, ensemble_predictions
# функции для работы с моделями: выбор основной модели, ансамблирование предсказаний и генерация ответа с использованием конвейера

from utils import check_model_files, load_main_model
# функции для проверки файлов модели и загрузки основной модели

from code_processing import separate_code_and_explanations, combine_code_and_translated_explanations
# функции для разделения кода и объяснений, а также их объединения после перевода

from resource_manager import enable_memory_growth, get_tokenizer_and_model, clear_model, get_models_list
# функции для управления ресурсами: включение роста памяти, получение токенизатора и модели, очистка модели, получение списка моделей

from text_proccessing import clean_text
# функция для очистки текста от лишних символов и форматирования

import translation
# функции для загрузки моделей перевода и перевода текста

from user_interface import user_interface
# функция для пользовательского интерфейса

init()  # Инициализация colorama для кроссплатформенного вывода цветного текста

# Определение используемого устройства (GPU или CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Работает on GPU:", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("Работает на CPU")

# Загрузка моделей перевода
translation_model, translation_tokenizer, back_translation_model, back_translation_token

# Загрузка моделей и токенизаторов для перевода и обратного перевода
translation_tokenizer = load_translation_models()


# Инициализация параметров моделей
models, tokenizers = [], []
weights = [1.0, 0.8]  # Веса для ансамблирования ответов от разных моделей
num_beams = 5  # Количество лучей для beam search при генерации ответов
temperature = 1.0  # Температура для генерации ответов (чем выше, тем более разнообразные ответы)

# Вывод информации о системе
print("system_info: CPU = {}, RAM = {} GB, OS = {}, Python = {}".format(
    platform.processor(),
    round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 2),
    platform.system(),
    sys.version
))

print("Загрузка моделей...")

# Загрузка основных моделей
for i in range(2):
    model_numbers = []
    models_list = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "rubert_cased_L-12_H-768_A-12_v2",
        "transfo-xl-wt103",
        "gpt-neo-2.7B",
    ]
    # model_number = input(f"Введите номер основной модели {i + 1}: ")
    # model_numbers.append(model_number)
    model_names = [models_list[int(model_number) - 1] for model_number in model_numbers]
    tokenizers, models = get_tokenizer_and_model(model_names)
    main_model_name, main_tokenizer_name = select_main_model(model_number)
    main_model, main_tokenizer = load_main_model(main_model_name, main_tokenizer_name)
    models.append(main_model)
    tokenizers.append(main_tokenizer)

print("Модели успешно загружены.")

print(Fore.GREEN + "Вопрос пользователя: " + Fore.RESET)
# Вывод текста вопроса пользователя

# Функция для обработки ввода пользователя, перевода вопроса, генерации ответа и обратного перевода
def process_user_input(
    user_prompt,
    models,
    tokenizers,
    translation_model,
    translation_tokenizer,
    back_translation_model,
    back_translation_tokenizer,
    weights,
    user_language="ru",  # добавить параметр языка ввода
    max_length=512,      # добавить параметр максимальной длины ввода
    max_retries=3,       # добавить параметр максимального количества попыток перевода или генерации текста
    num_beams=5,
    temperature=1.0
):
    if len(user_prompt) > max_length:
        print(f"Длина ввода слишком большая. Максимальная длина: {max_length}")
        return ""

    retries = 0
    while retries < max_retries:
        try:
            # Перевод вопроса пользователя на английский язык
            translated_prompt = translate_text(user_prompt, translation_model, translation_tokenizer, target_language="en")

            # Генерация ответа моделями
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

            # Ансамблирование ответов от разных моделей с использованием весов
            ensemble_response = ensemble_predictions(generated_responses, weights)

            # Обратный перевод сгенерированного ответа на русский язык
            translated_response = translate_text(ensemble_response, back_translation_model, back_translation_tokenizer, target_language="ru")

            # Возвращение переведенного ответа
            return translated_response

        except Exception as e:
            print("Ошибка во время генерации ответа. Повторяю...")
            retries += 1

    print("Не удалось сгенерировать ответ. Попробуйте еще раз.")
    return ""

# Главный цикл программы
while True:
    user_prompt = input("Введите ваш вопрос (или 'exit' для выхода): ")

    if user_prompt.lower() == "exit":
        break

    # Обработка ввода пользователя и получение ответа
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
    # Включаем режим роста памяти для TensorFlow, чтобы избежать ошибок из-за ограничения памяти
    enable_memory_growth()

    # Выбираем устройство для выполнения (GPU, если доступно, иначе CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Запускаем пользовательский интерфейс с передачей аргумента device
    user_interface(device)

    # Запускаем главную функцию
    main()

