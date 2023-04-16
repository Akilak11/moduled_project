# модуль config
import torch
import os
import pathlib
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_PATH = pathlib.Path("/mnt/c/Python_project/moduled_project/models")

# Переменная MODEL_NAMES должна быть выше MODEL_PATHS - потому-что она должна уже учавствует в определении MODEL_PATHS
MODEL_NAMES = {
    "pretrained": [
        {"bigcode": ["gpt_bigcode-santacoder"]},
        {"EleutherAI": ["gpt-neo-2.7B"]},
        "bert-base-uncased",
        "google_mt5-small",
        "google_mt5-xxl",
        "gpt2",
        {"Helsinki-NLP": ["opus-mt-en-ru", "opus-mt-ru-en"]},
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

# Переменная MODEL_NAMES должна быть выше MODEL_PATHS - потому-что она должна уже учавствует в определении MODEL_PATHS
PRETRAINED_MODELS_PATH = MODELS_PATH / "pretrain_models"
MYTRAINED_MODELS_PATH = MODELS_PATH / "mytrain_models"

PRETRAINED_MODEL_PATHS = {
    model_name: (PRETRAINED_MODELS_PATH.joinpath(model_name) if key is None else PRETRAINED_MODELS_PATH.joinpath(key, model_name))
    for model_name_dict in MODEL_NAMES["pretrained"]
    for key, model_name_list in (model_name_dict.items() if isinstance(model_name_dict, dict) else [(None, [model_name_dict])])
    for model_name in model_name_list
}

MYTRAINED_MODEL_PATHS = {
    model_name: MYTRAINED_MODELS_PATH.joinpath(model_name) for model_name in MODEL_NAMES["mytrained"]
}

MODELS_URL = "https://huggingface.co"

TRANSLATION_MODEL_NAME = "wmt19-ru-en"
BACK_TRANSLATION_MODEL_NAME = "wmt19-en-ru"

#Функция придающая веса по формуле Основная модель весит больше - 0.7, следующие 2 по 0.3, а остальные по формуле уменьшения весов последующих, от последнего веса константы на 30%. То есть по сути цикл - уменьшения весов моделей. Верхняя модель в списке - основная, чем ниже, тем меньше вес.
def calculate_weights(num_models, main_weight=0.7, secondary_weights=[0.3, 0.3], decay_rate=0.3):
    weights = [main_weight] + secondary_weights
    if num_models <= len(weights):
        return weights[:num_models]

    for _ in range(num_models - len(weights)):
        last_weight = weights[-1]
        new_weight = last_weight * (1 - decay_rate)
        weights.append(new_weight)

    return weights

num_models = len(MODEL_NAMES["pretrained"]) + len(MODEL_NAMES["mytrained"])
  # замените 'models' на список ваших моделей
weights = calculate_weights(num_models)

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
        "description": "Размер пакета данных при обучении модели",
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
    },
    {   "name": "WEIGHTS",
        "description": "Веса моделей в ансамбле предсказаний, не путать с весами внутри моделей",
        "default_value": weights,
        "applicable_models": ["all"]
    },
    {   "name": "MAX_MODELS_COUNT",
        "description": "Максимальное количество моделей для использования",
        "default_value": 3,
        "applicable_models": ["all"]
    },
    {   "name": "MODELS_COUNT",
        "description": "Количество моделей для использования",
        "default_value": 1,
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