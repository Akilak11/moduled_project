#модуль resource_manager.py
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM

model_cache = {}

#Функция get_models_list возвращает список имен моделей, доступных для использования в проекте.
def get_models_list():
    return [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "rubert_cased_L-12_H-768_A-12_v2",
        "transfo-xl-wt103",
        "gpt-neo-2.7B",
    ]
    
#Функция enable_memory_growth настраивает TensorFlow для роста памяти по мере необходимости. Это нужно, чтобы избежать ошибок из-за ограничения памяти.
def enable_memory_growth():
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

#Функция get_tokenizer_and_model принимает список имен моделей и загружает токенизаторы и модели для каждого имени из Hugging Face Transformers.
#Если токенизатор и модель уже были загружены, они извлекаются из кэша моделей вместо повторной загрузки.
def get_tokenizer_and_model(model_names):
    tokenizers = []
    models = []
    for model_name in model_names:
        if model_name in model_cache:
            tokenizer, model = model_cache[model_name]
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model_cache[model_name] = (tokenizer, model)
        tokenizers.append(tokenizer)
        models.append(model)
    return tokenizers, models

#Функция clear_models принимает список моделей и очищает их из памяти.
def clear_model(models):
    for model in models:
        del model

#Команда torch.cuda.empty_cache() используется для очистки кэша памяти на GPU.
torch.cuda.empty_cache()  # Если вы используете GPU
