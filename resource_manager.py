import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM

model_cache = {}

def enable_memory_growth():
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

def get_tokenizer_and_model(model_name):
    if model_name in model_cache:
        return model_cache[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_cache[model_name] = (tokenizer, model)
    return tokenizer, model

def clear_model(model):
    del model
    torch.cuda.empty_cache()  # Если вы используете GPU
