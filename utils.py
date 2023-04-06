# модуль utils
import os
import requests
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from config import MODELS_PATH, ENCODER_MODEL_NAME, DECODER_MODEL_NAME, MODELS_URL

def download_file(url, target_path):
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

def check_file_size(file_path):
    return file_path.stat().st_size

def is_file_available(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def download_models(models_path, model_names, base_url):
    if not models_path.exists():
        models_path.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        model_dir = models_path / model_name
        if not model_dir.exists():
            model_dir.mkdir()

        files_to_download = ["config.json", "pytorch_model.bin", "vocab.json", "merges.txt", "tokenizer_config.json"]

        with ThreadPoolExecutor(max_workers=5) as executor:
            for file in files_to_download:
                url = f"{base_url}/{model_name}/{file}"
                target_path = model_dir / file

                if not target_path.exists() or (is_file_available(url) and check_file_size(target_path) != int(requests.head(url).headers.get("Content-Length", 0))):
                    print(f"Downloading {url} to {target_path}")
                    executor.submit(download_file, url, target_path)

def setup_models():
    model_names = [ENCODER_MODEL_NAME, DECODER_MODEL_NAME]
    download_models(MODELS_PATH, model_names, MODELS_URL)

if __name__ == "__main__":
    setup_models()
