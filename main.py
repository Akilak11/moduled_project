from translation import load_translation_models, translate_text
from models import select_main_model, ensemble_predictions, generate_response
from utils import check_model_files, clean_text

# В функции process_user_input() добавьте обработку ошибок и ограничения на длину текста
def process_user_input(
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
            # Translate user input to English
            translated_prompt = translate_text(
                translation_model,
                translation_tokenizer,
                user_prompt,
                target_language="en",
                src_language=user_language,
            )

            # Generate a response in English
            generated_response = generate_response(
                models,
                [main_tokenizer] * len(models),  # используем один и тот же токенизатор для всех моделей
                translated_prompt,
                weights,
                num_beams=num_beams,
                temperature=temperature,
            )

            # Translate the response back to the user's language
            final_response = translate_text(
                back_translation_model,
                back_translation_tokenizer,
                generated_response,
                target_language=user_language,
                src_language="en",
            )

            return final_response
        except Exception as e:
            retries += 1
            print(f"Произошла ошибка: {e}. Попытка {retries} из {max_retries}")

    print(f"Не удалось обработать текст. Пожалуйста, попробуйте еще раз.")
    return ""

def main():
    translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer = load_translation_models()
    
    model_number = input("Введите номер основной модели: ")
    main_model_name, main_tokenizer = select_main_model(model_number)

    if main_model_name is not None:
        model_directory = '/mnt/c/Python_project/'  # Указываем путь к директории с моделями на вашем жестком диске

        if not check_model_files(main_model_name, model_directory):
            print(f"Модель {main_model_name} не прошла проверку. Пожалуйста, проверьте файлы модели.")
            return

        main_model = AutoModelForCausalLM.from_pretrained(os.path.join(model_directory, main_model_name)).to(device)

        while True:
            user_prompt = input("Введите текст: ")
            if user_prompt.lower() == "выход":
                break
            
            cleaned_user_prompt = clean_text(user_prompt)  # Очищаем введенный текст
            response = process_user_input(
                cleaned_user_prompt,  # Передаем очищенный текст
            # ... (остальные аргументы функции process_user_input)
        )
        cleaned_response = clean_text(response)  # Очищаем сгенерированный текст
        print(f"Ответ: {cleaned_response}")
            
            if user_prompt.lower() == "сохранить модели":
                directory = input("Введите директорию для сохранения: ")
                save_models_and_tokenizers(
                    main_model, main_tokenizer,
                    translation_model, translation_tokenizer,
                    back_translation_model, back_translation_tokenizer,
                    directory
                )
                print("Модели сохранены.")
                continue

            if user_prompt.lower() == "загрузить модели":
                directory = input("Введите директорию для загрузки: ")
                main_model, main_tokenizer, translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer = load_models_and_tokenizers(directory)
                print("Модели загружены.")
                continue

            response = process_user_input(
                user_prompt,
                main_model,
                main_tokenizer,
                translation_model,
                translation_tokenizer,
                back_translation_model,
                back_translation_tokenizer,
            )
            print(f"Ответ: {response}")
    else:
        print("Не выбран номер модели. Повторите попытку.")

if __name__ == "__main__":
    main()