#модуль generate_response.py
from transformers import pipeline
from load_models import load_main_model
from text_processing import clean_text, validate_input
from text_generator import TextGenerator
from translation_models import TranslationModel
from translation import TranslationService

# Создание экземпляров TranslationService, TextGenerator и TranslationModel
translation_service = TranslationService("Helsinki-NLP/opus-mt-en-ru", "cuda")
text_generator_instance = TextGenerator("Your_Encoder_Model_Name", "Your_Decoder_Model_Name", "cuda")
back_translation_model = TranslationModel(BACK_TRANSLATION_MODEL_NAME, DEVICE)

def generate_response(user_prompt: str, model_name: str, ensemble: bool = False,
                      back_translate: bool = False, num_beams=5, temperature=1.0,
                      use_encoder_decoder: bool = False, **kwargs):
    # Проверка ввода
    if not validate_input(user_prompt):
        return "Пожалуйста, введите корректный текст."

    # Очистка текста
    cleaned_text = clean_text(user_prompt)

    # Обратный перевод
    if back_translate:
        cleaned_text = back_translation_model.back_translate(cleaned_text)  # Использование экземпляра TranslationModel

    # Генерация ответа
    if use_encoder_decoder:
        response = text_generator_instance(cleaned_text)  # Использование экземпляра TextGenerator
    else:
        response = generate_response_with_pipeline(model_name, cleaned_text, num_beams, temperature, **kwargs)

    # Перевод ответа
    if back_translate:
        response = translation_service(response, max_length=512)  # Использование экземпляра TranslationService

    # Ансамбль предсказаний
    if ensemble:
        predictions = [response]
        weights = [1.0]

        if back_translate:
            translated_response = translation_service(response, max_length=512)  # Использование экземпляра TranslationService
            predictions.append(translated_response)
            weights.append(0.5)

        if use_encoder_decoder:
            encoder_decoder_response = text_generator_instance(clean_text(response))  # Использование экземпляра TextGenerator
            predictions.append(encoder_decoder_response)
            weights.append(0.5)

        response = ensemble_predictions(predictions, weights)

    return response

def generate_response_with_pipeline(model_name, user_prompt, num_beams=5, temperature=1.0, **kwargs):
    model, tokenizer = load_main_model(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0, num_beams=num_beams, max_length=1024, temperature=temperature, **kwargs)
    response = generator(user_prompt)[0]['generated_text']
    return response

def ensemble_predictions(predictions, weights):
    if len(predictions) != len(weights):
        raise ValueError("Количество предсказаний и весов должно совпадать")

    for weight in weights:
        if weight < 0:
            raise ValueError("Веса должны быть неотрицательными")

    ensemble_prediction = sum(prediction
