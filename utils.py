# модуль utils.py 
import torch
import os
import requests
import threading
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoTokenizer

from config import DEVICE, MAIN_MODEL, MODEL_PATHS, ENCODER_MODEL_NAME, DECODER_MODEL_NAME, TRANSLATION_MODEL_NAME, BACK_TRANSLATION_MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = MODELS_PATH / "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# Функция check_model_files принимает имя модели и директорию, в которой хранятся файлы модели, и проверяет, доступны ли эти файлы на сервере Hugging Face и имеют ли они такой же размер, как и локальные файлы на диске. 
# Если проверка не пройдена, функция возвращает False, в противном случае - True.
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
        return True

    local_config_size = os.path.getsize(local_config_path)
    local_model_size = os.path.getsize(local_model_path)

    if local_config_size != config_size or local_model_size != model_size:
        print("Размеры файлов моделей на жестком диске и сервере не совпадают.")
        return False

    return True

def download_file(url, local_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_model(model_name, model_directory):
    config_url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
    model_url = f"https://huggingface.co/{model_name}/resolve/main/pytorch_model.bin"

    local_config_path = os.path.join(model_directory, "config.json")
    local_model_path = os.path.join(model_directory, "pytorch_model.bin")

    print("Скачивание файлов модели...")

    thread1 = threading.Thread(target=download_file, args=(config_url, local_config_path))
    thread2 = threading.Thread(target=download_file, args=(model_url, local_model_path))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    print("Файлы модели успешно скачаны.")
