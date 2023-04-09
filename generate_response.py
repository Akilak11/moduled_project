# модуль generate_response
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline
from load_models import load_models, load_translation_models, load_pretrained_models
from text_processing import clean_text, validate_input, extract_code_and_explanations
from translation import ForwardTranslationService, BackTranslationService
from translation_models import TranslationModel
import config

# Загрузка моделей и токенизаторов
models, tokenizers = load_pretrained_models()
print("Модели загруженные в load_pretrained_models в generate_response: ", models)
print("Токенизеры загруженные в load_pretrained_models в generate_response: ", tokenizers)

translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer = load_translation_models()

# Создание экземпляров TranslationService
forward_translation_service = ForwardTranslationService(config.TRANSLATION_MODEL_NAME, config.DEVICE)
back_translation_service = BackTranslationService(config.TRANSLATION_MODEL_NAME, config.DEVICE)

def generate_response(user_prompt: str, ensemble: bool = False, back_translate: bool = False, weights=None):
    # Проверка ввода
    if not validate_input(user_prompt):
        return "Пожалуйста, введите корректный текст."

    # Очистка текста
    cleaned_text = clean_text(user_prompt)
    
    print("Очистка текста ", cleaned_text)

    # Обработка кода и комментариев во входящем тексте
    code_lines, explanations = separate_code_and_explanations(cleaned_text)
    translated_explanations = [translation_service.translate_text(explanation, back_translation_model) for explanation in explanations]
    cleaned_text = combine_code_and_translated_explanations(code_lines, translated_explanations)

    # Обратный перевод
    if back_translate:
        cleaned_text = translation_service.translate_text(cleaned_text, back_translation_model)
        print("Обратный перевод ", cleaned_text)
        
    # Генерация ответа
    if ensemble:
        if weights and len(weights) == 2:
            response_1 = generate_response_with_beam_search(model, tokenizer, cleaned_text)
            response_2 = generate_response_with_sampling(model, tokenizer, cleaned_text)
            
            if responses_1:
                print("Responses_1 exist")
            else:
                print("Responses_1 ensemble_prediction do not exist")
            
            
            if responses2:
                print("Responses_2 ensemble_prediction exist")
            else:
                print("Responses_2 ensemble_prediction do not exist")
            
            ensemble_prediction = ensemble_predictions([response_1, response_2], weights)
            
            if ensemble_prediction:
                print("ensemble_prediction_prediction exist")
            else:
                print("ensemble_prediction do not exist")
            
            response = ensemble_prediction

            if responses:
                print("Responses ensemble_prediction exist")
            else:
                print("Responses ensemble_prediction do not exist")
                print("responses are: ", responses)
                print(f"Responses count: {len(responses)}")

            if settings['weights']:
                print("settings['weights'] exist")
            else:
                print("settings['weights'] does not exist")
                print((settings['weights']))
                print(f"Weights count: {len(settings['weights'])}")
            if responses:
                print("The Numpy responses is : ")
                for i in responses:
                    print(i, end = ' ')
            if settings['weights']:
                print("The Numpy (settings['weights']) is: ")
                for i in (settings['weights']): 
                    print(i, end = " ")
            
                print(f"Responses count: {len(responses)}")
                print(f"Weights count: {len(settings['weights'])}")

        else:
            response = generate_response_with_pipeline(model, tokenizer, cleaned_text)
            
            if responses:
                print("Responses exist")
            else:
                print("Responses do not exist")
                print("responses are: ", responses)
                print(f"Responses count: {len(responses)}")

            if settings['weights']:
                print("settings['weights'] exist")
            else:
                print("settings['weights'] does not exist")
                print((settings['weights']))
                print(f"Weights count: {len(settings['weights'])}")
            if responses:
                print("The Numpy responses is : ")
                for i in responses:
                    print(i, end = ' ')
                print(f"Responses count: {len(responses)}")
                print(f"Weights count: {len(settings['weights'])}")

    else:
        response = generate_response_with_pipeline(model, tokenizer, cleaned_text)
        
        if responses:
                print("Responses exist")
        else:
            print("Responses do not exist")
            print("responses are: ", responses)
            print(f"Responses count: {len(responses)}")

        
        
        if responses:
            print("The Numpy responses is : ")
            for i in responses:
                print(i, end = ' ')
            
        print(f"Responses count: {len(responses)}")
        print(f"Weights count: {len(settings['weights'])}")
        
    # Обработка кода и комментариев в сгенерированном ответе
    code_lines, explanations = separate_code_and_explanations(response)
    translated_explanations = [translation_service.translate_text(explanation, translation_model) for explanation in explanations]
    response = combine_code_and_translated_explanations(code_lines, translated_explanations)

    # Перевод ответа
    if back_translate:
        response = translation_service.translate_text(response, translation_model, reverse=True)

    return response

    

def generate_response_with_pipeline(model, tokenizer, user_prompt):
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=config.DEVICE, num_beams=config.NUM_BEAMS, max_length=config.MAX_LENGTH, temperature=config.TEMPERATURE)
    response = generator(user_prompt)[0]['generated_text']
    
    if responses:
                print("Responses with_pipeline exist")
    else:
        print("Responses with_pipeline do not exist")
        print("responses are: ", responses)
        print(f"Responses count: {len(responses)}")
                
    return response

def generate_response_with_beam_search(model, tokenizer, user_prompt):
    input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(config.DEVICE)
    print("Input IDs: ", input_ids)
    output = model.generate(
        input_ids,
        max_length=config.MAX_LENGTH,
        num_beams=config.NUM_BEAMS,
        temperature=config.TEMPERATURE,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    if responses:
                print("Responses with beam search exist")
    else:
        print("Responses with_beam_search do not exist")
        print("responses are: ", responses)
        print(f"Responses count: {len(responses)}")
                
    return response

def generate_response_with_sampling(model, tokenizer, user_prompt):
    input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(DEVICE)
    print("Input IDs: ", input_ids)
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
    if responses:
                print("Responses with_sampling exist")
    else:
        print("Responses with_sampling do not exist")
        print("responses are: ", responses)
        print(f"Responses count: {len(responses)}")
    return response

def ensemble_predictions(predictions, weights):
    if len(predictions) != len(weights):
        raise ValueError("Количество предсказаний и весов должно совпадать")

    for weight in weights:
        if weight < 0:
            raise ValueError("Веса должны быть неотрицательными")

    ensemble_prediction = sum(prediction * weight for prediction, weight in zip(predictions, weights)) / sum(weights)
    return ensemble_prediction
