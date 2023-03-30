import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM

model_cache = {}


def get_models_list():
    return [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "rubert_cased_L-12_H-768_A-12_v2",
        "transfo-xl-wt103",
        "gpt-neo-2.7B",
    ]


def enable_memory_growth():
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

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

def clear_models(models):
    for model in models:
        del model
        
torch.cuda.empty_cache()  # Если вы используете GPU
