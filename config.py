# модуль config
import torch
import os
import pathlib
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_PATH = pathlib.Path("/mnt/c/Python_project/moduled_project/models")

MODEL_NAMES = {
    "pretrained": [
    "EleutherAI/gpt-neo-2.7B",
    "bert-base-uncased",
    "google_mt5-small",
    "google_mt5-xxl",
    "gpt2",
    "Helsinki-NLP/opus-mt-en-ru",
    "Helsinki-NLP/opus-mt-ru-en",
    "roberta-base",
    "rubert_cased_L-12_H-768_A-12_v2",
    "sberbank-ai_ruclip-vit-large-patch14-336",
    "transfo-xl-wt103",
    "wmt19-en-ru",
    "wmt19-ru-en",
    ],
    "mytrained": [
        # ... (названия ваших собственных обученных моделей)
    ]
}

MODEL_PATHS = {
    model_name: MODELS_PATH / "pretrain_models" / model_name for model_name in MODEL_NAMES["pretrained"]
}
MODEL_PATHS.update({
    model_name: MODELS_PATH / "mytrain_models" / model_name for model_name in MODEL_NAMES["mytrained"]
})

MODELS_URL = "https://huggingface.co"

TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"
BACK_TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"

PARAMETERS = [
    {
        "name": "TEMPERATURE",
        "description": "Температура генерации текста",
        "default_value": 1.0,
        "applicable_models": ["all"]
    },
    {
        "name": "MAX_LENGTH",
        "description": "Максимальная длина ответа",
        "default_value": 512,
        "applicable_models": ["all"]
    },
    {
        "name": "MIN_LENGTH",
        "description": "Минимальная длина ответа",
        "default_value": 1,
        "applicable_models": ["all"]
    },
    {
        "name": "TOP_K",
        "description": "Количество верхних кандидатов для сэмплирования",
        "default_value": 50,
        "applicable_models": ["all"]
    },
    {
        "name": "NUM_BEAMS",
        "description": "Количество лучей для поиска лучшего пути",
        "default_value": 5,
        "applicable_models": ["all"]
    },
    {   "name": "BATCH_SIZE",
        "description": "Размер пакета данных при обучении модел",
        "default_value": 32,
        "applicable_models": ["all"]
    },
    {   "name": "EPOCHS",
        "description": "Количество эпох обучения модели",
        "default_value": 10,
        "applicable_models": ["all"]
    },
    {   "name": "LEARNING_RATE",
        "description": "Cкорость обучения модели",
        "default_value": 0.001,
        "applicable_models": ["all"]
    },
    {   "name": "NUM_CLASSES",
        "description": "Количество классов при обучении модели",
        "default_value": 32,
        "applicable_models": ["all"]
    },
    {   "name": "INPUT_SHAPE",
        "description": "Форма входных данных при обучении модели",
        "default_value": (28, 28, 1),
        "applicable_models": ["all"]
    },
    {   "name": "NUM_BEAMS_GROUP",
        "description": "Количество групп лучей при генерации текста. Каждая группа представляет собой набор лучей с одинаковыми начальными символами.",
        "default_value": 5,
        "applicable_models": ["all"]
    }
]

# Encoder и Decoder модели
ENCODER_MODEL_NAME = "encoder_model"
DECODER_MODEL_NAME = "decoder_model"

# Пути к данным
DATA_PATH = Path("/mnt/c/Python_project/data")
TRAIN_DATA_PATH = DATA_PATH / "train.csv"
TEST_DATA_PATH = DATA_PATH / "test.csv"

NUM_WORKERS = os.cpu_count()

# Print information
print(f"Using device: {DEVICE}")
print(f"Number of workers: {NUM_WORKERS}")