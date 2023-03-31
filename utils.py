import os
import re
import requests

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

#Функция load_main_model принимает имя основной модели и имя токенизатора, и загружает модель и токенизатор с помощью Hugging Face Transformers.
#Модель затем отправляется на устройство, доступное для выполнения (GPU или CPU).
def load_main_model(main_model_name, main_tokenizer_name):
    main_model = AutoModelForCausalLM.from_pretrained(main_model_name).to(device)
    main_tokenizer = AutoTokenizer.from_pretrained(main_tokenizer_name)
    return main_model, main_tokenizer