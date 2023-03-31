import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Используйте функцию select_main_model() для выбора GPT-3.5 или GPT-4 модели
def select_main_model(model_number):
    models = {
        "1": ("bert-base-uncased", "Bert"),
        "2": ("roberta-base", "Roberta"),
        "3": ("gpt2", "GPT2"),
        "4": ("rubert_cased_L-12_H-768_A-12_v2", "RuBert"),
        "5": ("transfo-xl-wt103", "TransformerXL"),
        "6": ("gpt-neo-2.7B", "EleutherAI")
        # Добавьте здесь строки для GPT-3.5 и GPT-4
    }

# Показываем какие модели используются в проекте
    print(models)

    model_name, model_class_name = models.get(str(model_number), (None, None))

    if model_name is None:
        print(f"Модель с номером {model_number} не найдена.")
        return None, None

    tokenizer_class = getattr(transformers, f"{model_class_name}Tokenizer")
    main_tokenizer = tokenizer_class.from_pretrained(model_name)
    main_tokenizer.pad_token = main_tokenizer.eos_token # Устанавливаем pad_token равным eos_token
    return model_name, main_tokenizer

# Загрузите GPT-3.5 и GPT-4 модели и включите их в список моделей
model1_name = "EleutherAI/gpt-neo-2.7B"  # Имя модели GPT-3.5
model1 = AutoModelForCausalLM.from_pretrained(model1_name).to(device)

model2_name, model2_tokenizer = select_main_model(2)
model2 = AutoModelForCausalLM.from_pretrained(model2_name).to(device)

model3_name, model3_tokenizer = select_main_model(3)
model3 = AutoModelForCausalLM.from_pretrained(model3_name).to(device)

models = [model1, model2, model3]
tokenizers = [model1_tokenizer, model2_tokenizer, model3_tokenizer] # Сохраняем токенизаторы в отдельном списке
weights = [0.1, 0.1, 0.5]

def ensemble_predictions(predictions, weights):
    if len(predictions) != len(weights):
        raise ValueError("Количество предсказаний и весов должно совпадать")

    for weight in weights:
        if weight < 0:
            raise ValueError("Веса должны быть неотрицательными")

    ensemble_prediction = sum(prediction * weight for prediction, weight in zip(predictions, weights)) / sum(weights)
    return ensemble_prediction
    
# В функции generate_response() добавьте параметры, такие как num_beams и temperature, для управления генерацией текста
def generate_response_with_pipeline(models, tokenizers, user_prompt, weights, num_beams=5, temperature=1.0, **kwargs):
    predictions = []

    for model, tokenizer, weight in zip(models, tokenizers, weights):
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0, num_beams=num_beams, max_length=1024, temperature=temperature * weight, **kwargs)
        prediction = generator(user_prompt)[0]['generated_text']
        predictions.append(prediction)

    response = max(set(predictions), key=predictions.count)
    return response
