# модуль generate_response.py
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoForCausalLM, GPT2Tokenizer
from load_models import load_models, load_translation_models
from text_processing import clean_text, validate_input
from text_generator import TextGenerator
from translation_models import TranslationModel
from translation import TranslationService
from config import ENCODER_MODEL_NAME, DECODER_MODEL_NAME, DEVICE, GENERATION_MAX_LENGTH, GENERATION_MIN_LENGTH, GENERATION_TEMPERATURE, TRANSLATION_MODEL_NAME, BACK_TRANSLATION_MODEL_NAME

text_generator_instance = TextGenerator(ENCODER_MODEL_NAME, DECODER_MODEL_NAME, DEVICE, GENERATION_MAX_LENGTH, GENERATION_MIN_LENGTH, GENERATION_TEMPERATURE)

# Создание экземпляров TranslationService, TextGenerator и TranslationModel
translation_service = TranslationService(TRANSLATION_MODEL_NAME, DEVICE)
back_translation_model = TranslationModel(BACK_TRANSLATION_MODEL_NAME, DEVICE)

def generate_response(user_prompt: str, model_names: list, ensemble: bool = False,
                      back_translate: bool = False, num_beams=5, temperature=1.0,
                      use_encoder_decoder_list=None, weights=None, **kwargs):
    # Проверка ввода
    if not validate_input(user_prompt):
        return "Пожалуйста, введите корректный текст."

    # Очистка текста
    cleaned_text = clean_text(user_prompt)

    # Обратный перевод
    if back_translate:
        cleaned_text = back_translation_model.back_translate(cleaned_text)  # Использование экземпляра TranslationModel

    # Генерация ответа
    if ensemble and len(model_names) > 1 and weights and len(model_names) == len(weights):
        responses = []
        for idx, model_name in enumerate(model_names):
            use_encoder_decoder = use_encoder_decoder_list[idx] if use_encoder_decoder_list else False
            if use_encoder_decoder:
                response = text_generator_instance.generate(cleaned_text)  # Использование экземпляра TextGenerator
            else:
                response = generate_response_with_pipeline(model_name, cleaned_text, num_beams, temperature, **kwargs)
            responses.append(response)
        response = ensemble_predictions(responses, weights)
    else:
        use_encoder_decoder = use_encoder_decoder_list[0] if use_encoder_decoder_list else False
        if use_encoder_decoder:
            response = text_generator_instance.generate(cleaned_text)  # Использование экземпляра TextGenerator
        else:
            response = generate_response_with_pipeline(model_names[0], cleaned_text, num_beams, temperature, **kwargs)

    # Перевод ответа
    if back_translate:
        response = translation_service.translate(response, max_length=512)  # Использование экземпляра TranslationService

    return response


def generate_response_with_pipeline(model_name, user_prompt, num_beams=5, temperature=1.0, **kwargs):
    model, tokenizer = load_models(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0, num_beams=num_beams, max_length=1024, temperature=temperature, **kwargs)
    response = generator(user_prompt)[0]['generated_text']
    return response

def ensemble_predictions(predictions, weights):
    if len(predictions) != len(weights):
        raise ValueError("Количество предсказаний и весов должно совпадать")

    for weight in weights:
        if weight < 0:
            raise ValueError("Веса должны быть неотрицательными")

    ensemble_prediction = sum(prediction * weight for prediction, weight in zip(predictions, weights)) / sum(weights)
    return ensemble_prediction

