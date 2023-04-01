from transformers import pipeline
from load_models import load_main_model
from text_processing import clean_text, validate_input
from text_generator import text_generator

def generate_response_with_pipeline(model_name, user_prompt, num_beams=5, temperature=1.0, **kwargs):
    model, tokenizer = load_main_model(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0, num_beams=num_beams, max_length=1024, temperature=temperature, **kwargs)
    response = generator(user_prompt)[0]['generated_text']
    return response

def generate_response(user_prompt: str, model_name: str, ensemble: bool = False,
                      back_translate: bool = False, num_beams=5, temperature=1.0,
                      use_encoder_decoder: bool = False, **kwargs):
    # Validate input
    if not validate_input(user_prompt):
        return "Пожалуйста, введите корректный текст."

    # Clean text
    cleaned_text = clean_text(user_prompt)

    if use_encoder_decoder:
        response = text_generator(cleaned_text)
    else:
        response = generate_response_with_pipeline(model_name, cleaned_text, num_beams, temperature, **kwargs)

    return response

def ensemble_predictions(predictions, weights):
    if len(predictions) != len(weights):
        raise ValueError("Количество предсказаний и весов должно совпадать")

    for weight in weights:
        if weight < 0:
            raise ValueError("Веса должны быть неотрицательными")

    ensemble_prediction = sum(prediction * weight for prediction, weight in zip(predictions, weights)) / sum(weights)
    return ensemble_prediction


'''
каким образом теперь лучше всего включить код с translation_models.py в generate_response.py
предложи обновленный вариант код, с улучшениями, но важно, без потери функциональности предыдущего, предложи готовый код с комментариями на русском языке

from transformers import pipeline
from load_models import load_main_model
from text_processing import clean_text, validate_input
from text_generator import text_generator

def generate_response_with_pipeline(model_name, user_prompt, num_beams=5, temperature=1.0, **kwargs):
    model, tokenizer = load_main_model(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0, num_beams=num_beams, max_length=1024, temperature=temperature, **kwargs)
    response = generator(user_prompt)[0]['generated_text']
    return response

def generate_response(user_prompt: str, model_name: str, ensemble: bool = False,
                      back_translate: bool = False, num_beams=5, temperature=1.0,
                      use_encoder_decoder: bool = False, **kwargs):
    # Validate input
    if not validate_input(user_prompt):
        return "Пожалуйста, введите корректный текст."

    # Clean text
    cleaned_text = clean_text(user_prompt)

    if use_encoder_decoder:
        response = text_generator(cleaned_text)
    else:
        response = generate_response_with_pipeline(model_name, cleaned_text, num_beams, temperature, **kwargs)

    return response

def ensemble_predictions(predictions, weights):
    if len(predictions) != len(weights):
        raise ValueError("Количество предсказаний и весов должно совпадать")

    for weight in weights:
        if weight < 0:
            raise ValueError("Веса должны быть неотрицательными")

    ensemble_prediction = sum(prediction * weight for prediction, weight in zip(predictions, weights)) / sum(weights)
    return ensemble_prediction

'''