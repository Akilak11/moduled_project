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
    while True:
        print("Выберите опцию:")
        print("1. Задать вопрос")
        print("2. Изменить настройки")
        print("3. Выйти")

        user_choice = input("Введите номер опции: ")

        if user_choice == "1":
            user_prompt = input("Введите ваш вопрос: ")
            user_prompt = clean_text(user_prompt)
            user_prompt = separate_code_and_explanations(user_prompt)
            user_prompt = combine_code_and_translated_explanations(user_prompt)

            # Вызов функции генерации ответа здесь (ваша функция, которая использует модель GPT)
            # Здесь предполагается, что вы добавите код для генерации ответа с помощью модели GPT.
            # Не забудьте использовать переменную device для определения, где будет запущена модель (на CPU или GPU).

            # Здесь выводится сгенерированный ответ
            print("Ответ:")
            print(answer)

        elif user_choice == "2":
            # Измените настройки, например, максимальную длину ответа или температуру
            pass

        elif user_choice == "3":
            print("Выход...")
            break

        else:
            print("Некорректный выбор, попробуйте еще раз.")