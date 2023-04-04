#модуль utils.py
import torch
import os
import re
import requests
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from config import main_model_path, DEVICE
from utils import check_model_files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Функция check_model_files принимает имя модели и директорию, в которой хранятся файлы модели, и проверяет, доступны ли эти файлы на сервере Hugging Face и имеют ли они такой же размер, как и локальные файлы на диске. 
#Если проверка не пройдена, функция возвращает False, в противном случае - True.
def check_model_files(model_name, model_directory):
    config_url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
    model_url = f"https://huggingface.co/{model_name}/resolve/main/pytorch_model.bin"

    response_config = requests.head(config_url)
    response_model = requests.head(model_url)

    if response_config.status_code != 200 or response_model.status_code != 200:
        print("Не удалось получить информацию о модели с сервера.")
        return False

    config_size = int(response_config.headers["Content-Length"])
    model_size = int(response_model.headers["Content-Length"])

    local_config_path = os.path.join(model_directory, "config.json")
    local_model_path = os.path.join(model_directory, "pytorch_model.bin")

    if not os.path.isfile(local_config_path) or not os.path.isfile(local_model_path):
        print("Один или оба файла модели отсутствуют на жестком диске.")
        return False

    local_config_size = os.path.getsize(local_config_path)
    local_model_size = os.path.getsize(local_model_path)

    if local_config_size != config_size or local_model_size != model_size:
        print("Размеры файлов моделей на жестком диске и сервере не совпадают.")
        return False

    return True
    
def load_main_model(model_name):
    model_directory = os.path.join("models", model_name)
    
    if not check_model_files(model_name, model_directory):
        print("Проверка файлов модели не пройдена. Пожалуйста, убедитесь, что файлы модели доступны и имеют правильный размер.")
        return

    model = GPTNeoForCausalLM.from_pretrained(model_directory).to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return model, tokenizer

def download_model(model_name, model_directory):
    config_url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
    model_url = f"https://huggingface.co/{model_name}/resolve/main/pytorch_model.bin"

    local_config_path = os.path.join(model_directory, "config.json")
    local_model_path = os.path.join(model_directory, "pytorch_model.bin")

    print("Скачивание файлов модели...")

    response_config = requests.get(config_url)
    with open(local_config_path, "wb") as f:
        f.write(response_config.content)

    response_model = requests.get(model_url)
    with open(local_model_path, "wb") as f:
        f.write(response_model.content)

    print("Файлы модели успешно скачаны.")
