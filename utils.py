# модуль utils
import json
import os
import requests
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlretrieve
from tqdm import tqdm
from typing import List, Dict, Union

import config
''' Изменил фукнцию надо спросить за неё
def get_model_files(base_url, model_name):
    model_name_in_url = model_name.replace("/", "-")
    url = f"{base_url}/{model_name_in_url}/resolve/main/"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            files = json.loads(response.text)
            return files
        else:
            print(f"Failed to get model files for {model_name}: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error getting model files for {model_name}: {e}")
        return []
'''
def get_model_files(base_url, model_name):
    model_name_in_url = model_name.replace("/", "-")
    url = f"{base_url}/{model_name_in_url}/resolve/main/config.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json"]
        else:
            return []
    except Exception as e:
        print(f"Error getting model files: {e}")
        return []

def download_file(url, target_path):

    download_models(target_dir, MODEL_NAMES["pretrained"], models_url)
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    filename = url.split("/")[-1]

    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
        temp_path = Path(temp_file.name)

        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc=filename, ncols=100
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                temp_file.write(data)
                progress_bar.update(len(data))

    os.rename(temp_path, target_path)
''' он изменил функцию
def download_file(url, target_path):
    try:
        response = requests.get(url, stream=True)
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        print(f"Error downloading file {url}: {e}")
'''

def check_file_size(file_path):
    return file_path.stat().st_size
    
''' он изменил эту функцию спросить за неё
def is_file_available(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
'''
def is_file_available(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except Exception as e:
        print(f"Error checking file availability: {e}")
        return False



''' Он изменил функцию, спросить за неё
def download_models(models_path, model_names, base_url):
    if not models_path.exists():
        models_path.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        model_dir = models_path / model_name
        if not model_dir.exists():
            model_dir.mkdir()

        files_to_download = get_model_files(base_url, model_name)

        if not files_to_download:  # Если список файлов пуст, пропустить модель
            continue

        for file in files_to_download:
            target_path = model_dir / file
            if not target_path.exists():
                with open(target_path, 'w') as empty_file:
                    pass

        with ThreadPoolExecutor(max_workers=5) as executor:
            for file in files_to_download:
                model_name_in_url = model_name.replace("/", "-")
                url = f"{base_url}/{model_name_in_url}/resolve/main/{file}"
                target_path = model_dir / file

                if not target_path.exists() or (is_file_available(url) and check_file_size(target_path) != int(requests.head(url).headers.get("Content-Length", 0))):
                    print(f"Downloading {url} to {target_path}")
                    executor.submit(download_file, url, target_path)
'''
def download_models(target_dir: Path, model_names: List[Union[str, Dict[str, List[str]]]], models_url: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        model_path = target_dir / model_name
        model_url = f"{models_url}/{model_name}"

        if is_file_available(model_path):
            print(f"Model {model_name} already exists in local storage. Skipping download.")
        else:
            print(f"Downloading {model_name} from {model_url} to {model_path}...")
            urlretrieve(model_url, model_path)
            print(f"{model_name} downloaded successfully.")

def is_model_available_locally(model_path):
    config_path = model_path / "config.json"
    model_weights_path = model_path / "pytorch_model.bin"
    return config_path.exists() and model_weights_path.exists()

'''Функция на будущее, для векторных моделей и обработок.
def setup_models():
    model_names = [ENCODER_MODEL_NAME, DECODER_MODEL_NAME]
    download_models(MODELS_PATH, model_names, MODELS_URL)

setup_models()
'''

# Вызовите функцию, чтобы начать загрузку моделей
#download_models()

'''    from transformers import AutoTokenizer, AutoModelForCausalLM
import config

# Для каждой модели в списке
for model_name, custom_cache_dir in config.PRETRAINED_MODEL_PATHS.items():
    try:
        print(f"Downloading and caching {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir)
        print(f"Successfully downloaded {model_name} to {custom_cache_dir}")
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")'''