# модуль generate_response
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoForCausalLM, GPT2Tokenizer
from transformers import pipeline
from load_models import load_models, load_translation_models
from text_processing import clean_text, validate_input, extract_code_and_explanations
from translation import TranslationService
from translation_models import TranslationModel
from config import PARAMETERS, MODEL_NAMES, BACK_TRANSLATION_MODEL_NAME, TRANSLATION_MODEL_NAME
from code_processing import separate_code_and_explanations, combine_code_and_translated_explanations

# Получение параметров из PARAMETERS
MODEL_NAME = MODEL_NAMES['model_name']
DEVICE = DEVICE['device']
TRANSLATION_MODEL_NAME = TRANSLATION_MODEL_NAME['translation_model_name']
BACK_TRANSLATION_MODEL_NAME = BACK_TRANSLATION_MODEL_NAME['back_translation_model_name']
MAX_LENGTH = PARAMETERS['max_length']
TEMPERATURE = PARAMETERS['temperature']
NUM_BEAMS = PARAMETERS['num_beams']

# Загрузка моделей и токенизаторов
model, tokenizer = load_models(PARAMETERS)
translation_model, back_translation_model = load_translation_models(PARAMETERS)

# Создание экземпляров TranslationService
translation_service = TranslationService(TRANSLATION_MODEL_NAME, DEVICE)

def generate_response(user_prompt: str, ensemble: bool = False, back_translate: bool = False, weights=None):
    # Проверка ввода
    if not validate_input(user_prompt):
        return "Пожалуйста, введите корректный текст."

    # Очистка текста
    cleaned_text = clean_text(user_prompt)

    # Обработка кода и комментариев во входящем тексте
    code_lines, explanations = separate_code_and_explanations(cleaned_text)
    translated_explanations = [translation_service.translate_text(explanation, back_translation_model) for explanation in explanations]
    cleaned_text = combine_code_and_translated_explanations(code_lines, translated_explanations)

    # Обратный перевод
    if back_translate:
        cleaned_text = translation_service.translate_text(cleaned_text, back_translation_model)

    # Генерация ответа
    if ensemble:
        if weights and len(weights) == 2:
            response_1 = generate_response_with_beam_search(model, tokenizer, cleaned_text)
            response_2 = generate_response_with_sampling(model, tokenizer, cleaned_text)

            ensemble_prediction = ensemble_predictions([response_1, response_2], weights)
            response = ensemble_prediction
        else:
            response = generate_response_with_pipeline(model, tokenizer, cleaned_text)
    else:
        response = generate_response_with_pipeline(model, tokenizer, cleaned_text)

    # Обработка кода и комментариев в сгенерированном ответе
    code_lines, explanations = separate_code_and_explanations(response)
    translated_explanations = [translation_service.translate_text(explanation, translation_model) for explanation in explanations]
    response = combine_code_and_translated_explanations(code_lines, translated_explanations)

    # Перевод ответа
    if back_translate:
        response = translation_service.translate_text(response, translation_model, reverse=True)

    return response

def generate_response_with_pipeline(model, tokenizer, user_prompt):
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=DEVICE, num_beams=NUM_BEAMS, max_length=MAX_LENGTH, temperature=TEMPERATURE)
    response = generator(user_prompt)[0]['generated_text']
    return response

def generate_response_with_beam_search(model, tokenizer, user_prompt):
    input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(
        input_ids,
        max_length=MAX_LENGTH,
        num_beams=NUM_BEAMS,
        temperature=TEMPERATURE,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def generate_response_with_sampling(model, tokenizer, user_prompt):
    input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(
        input_ids,
        max_length=MAX_LENGTH,
        do_sample=True,
        temperature=TEMPERATURE,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def ensemble_predictions(predictions, weights):
    if len(predictions) != len(weights):
        raise ValueError("Количество предсказаний и весов должно совпадать")

    for weight in weights:
        if weight < 0:
            raise ValueError("Веса должны быть неотрицательными")

    ensemble_prediction = sum(prediction * weight for prediction, weight in zip(predictions, weights)) / sum(weights)
    return ensemble_prediction

