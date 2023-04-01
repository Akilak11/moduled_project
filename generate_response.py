from transformers import pipeline
from load_models import load_main_model


# Используйте функцию select_main_model() для выбора GPT-3.5 или GPT-4 модели
models = {
    "1": "bert-base-uncased",
    "2": "roberta-base",
    "3": "gpt2",
    "4": "rubert_cased_L-12_H-768_A-12_v2",
    "5": "transfo-xl-wt103",
    "6": "gpt-neo-2.7B",
    # Добавьте здесь строки для GPT-3.5 и GPT-4
}

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

    ensemble_prediction = sum(prediction * weight for prediction, weight in zip(predictions, weights)) / sum(weights)
    return ensemble_prediction

#---------------------------------------------------------------------------------------------------------------
from typing import List
import torch
from config import GENERATE_RESPONSE_MAX_LENGTH, DEVICE, ANSWER_MODEL_1_PATH, ANSWER_MODEL_2_PATH, ANSWER_MODEL_3_PATH, ENSEMBLE_WEIGHTS
from models import AnswerModelEnsemble
from text_processing import clean_text
from translation import TranslationService


def process_text(text: str, translation_service: TranslationService, back_translate: bool = False) -> str:
    # Clean text
    cleaned_text = clean_text(text)

    # Translate text to English
    translated_text = translation_service(cleaned_text, GENERATE_RESPONSE_MAX_LENGTH)

    # If back translation is enabled, perform back-translation
    if back_translate:
        back_translated_text = translation_service.translate(translated_text, GENERATE_RESPONSE_MAX_LENGTH)
        cleaned_back_translated_text = clean_text(back_translated_text)
        inputs = [cleaned_text.lower(), cleaned_back_translated_text.lower()]
    else:
        inputs = [cleaned_text.lower()]

    # Load ensemble model
    model_1 = torch.load(ANSWER_MODEL_1_PATH)
    model_2 = torch.load(ANSWER_MODEL_2_PATH)
    model_3 = torch.load(ANSWER_MODEL_3_PATH)
    models = [model_1, model_2, model_3]
    ensemble = AnswerModelEnsemble(models, ENSEMBLE_WEIGHTS)

    # Generate response
    with torch.no_grad():
        response = ensemble.generate(inputs, GENERATE_RESPONSE_MAX_LENGTH, DEVICE)

    return response[0]
