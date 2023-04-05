# config.pyimport os
import torch
import re
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Конфигурация для моделей
MAIN_MODEL = "EleutherAI/gpt-neo-2.7B"
ENCODER_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
DECODER_MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"
GENERATION_MAX_LENGTH = 100
GENERATION_MIN_LENGTH = 10
GENERATION_TEMPERATURE = 0.7

#Конфигурация для других настроек
TRANSLATION_MAX_LENGTH = 512
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"
BACK_TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"

# Директории
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = Path("/mnt/c/Python_project/models")
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrain_models"
MYTRAIN_MODELS_DIR = MODELS_DIR / "mytrain_models"

# Пути к данным
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

# Параметры модели
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

# Обновленные пути к моделям
MODEL_PATHS = {
    "main_model": PRETRAINED_MODELS_DIR / "gpt-neo-2.7B",
    "translation_model": PRETRAINED_MODELS_DIR / "Helsinki-NLP/opus-mt-en-ru",
    "back_translation_model": PRETRAINED_MODELS_DIR / "Helsinki-NLP/opus-mt-ru-en",
    "answer_model_1": MYTRAIN_MODELS_DIR / "answer_model_1",
    "answer_model_2": MYTRAIN_MODELS_DIR / "answer_model_2",
    "answer_model_3": MYTRAIN_MODELS_DIR / "answer_model_3",
    "model1": MYTRAIN_MODELS_DIR / "model1",
    "model2": MYTRAIN_MODELS_DIR / "model2",
    "model3": MYTRAIN_MODELS_DIR / "model3",
}
#weights = [0.4, 0.3, 0.3]

'''# Новые пути к моделям (для будущих моделей)
MODEL_PATHS = {
    "model1": MYTRAIN_MODELS_DIR / "model1",
    "model2": MYTRAIN_MODELS_DIR / "model2",
    "model3": MYTRAIN_MODELS_DIR / "model3"
}'''

# Hardware parameters
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
NUM_WORKERS = os.cpu_count()

# Пути к предварительно обученным моделям
GPT2_MODEL_PATH = PRETRAINED_MODELS_DIR / "gpt2"
MT_RU_EN_MODEL_PATH = PRETRAINED_MODELS_DIR / "Helsinki-NLP/opus-mt-ru-en"
MT_EN_RU_MODEL_PATH = PRETRAINED_MODELS_DIR / "Helsinki-NLP/opus-mt-en-ru"
RUBERT_MODEL_PATH = PRETRAINED_MODELS_DIR / "rubert_cased_L-12_H-768_A-12_v2"
ROBERTA_BASE_MODEL_PATH = PRETRAINED_MODELS_DIR / "roberta-base"


MODEL_NAMES = model_names
model_names = [
    "gpt-neo-2.7B"
    "bert-base-uncased",
    "google_mt5-small",
    "google_mt5-xxl",
    "gpt2",
    "Helsinki-NLP/opus-mt-ru-en",
    "Helsinki-NLP/opus-mt-en-ru",
    "roberta-base",
    "rubert_cased_L-12_H-768_A-12_v2",
    "sberbank-ai_ruclip-vit-large-patch14-336",
    "transfo-xl-wt103",
    "wmt19-en-ru",
    "wmt19-ru-en",
    ]

models = []
tokenizers = []

# Print information
print(f"Using device: {DEVICE}")
print(f"Number of workers: {NUM_WORKERS}")

'''Примеры других config.py
-------------------------------------------------------------------------------------------------------------------------------------------------
import configparser

# Создаем объект ConfigParser
config = configparser.ConfigParser()

# Читаем конфигурационный файл
config.read('config.ini')

# Получаем значения из секции 'paths'
TRAIN_DATA_PATH = config.get('paths', 'TRAIN_DATA_PATH')
TEST_DATA_PATH = config.get('paths', 'TEST_DATA_PATH')
MODELS_PATH = config.get('paths', 'MODELS_PATH')
TRANSLATION_MODEL_PATH = config.get('paths', 'TRANSLATION_MODEL_PATH')
BACK_TRANSLATION_MODEL_PATH = config.get('paths', 'BACK_TRANSLATION_MODEL_PATH')

# Получаем значения из секции 'models'
ENSEMBLE_MODEL_1 = config.get('models', 'ENSEMBLE_MODEL_1')
ENSEMBLE_MODEL_2 = config.get('models', 'ENSEMBLE_MODEL_2')
ENSEMBLE_MODEL_3 = config.get('models', 'ENSEMBLE_MODEL_3')



-------------------------------------------------------------------------------------------------------------------------------------------------
import os

# Директория с данными
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Пути к данным
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.csv')

# Директория с моделями
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Пути к моделям
MAIN_MODEL_PATH = os.path.join(MODELS_DIR, 'main_model')
TRANSLATION_MODEL_PATH = os.path.join(MODELS_DIR, 'translation_model')
BACK_TRANSLATION_MODEL_PATH = os.path.join(MODELS_DIR, 'back_translation_model')

# Гиперпараметры
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


-------------------------------------------------------------------------------------------------------------------------------------------------
import os
from pathlib import Path

# Директории
ROOT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = ROOT_DIR / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"

# Файлы данных
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

# Модели машинного обучения
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"
BACK_TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
ANSWER_MODEL_1_PATH = str(TRAINED_MODELS_DIR / "answer_model_1")
ANSWER_MODEL_2_PATH = str(TRAINED_MODELS_DIR / "answer_model_2")
ANSWER_MODEL_3_PATH = str(TRAINED_MODELS_DIR / "answer_model_3")
ENSEMBLE_WEIGHTS = [0.4, 0.3, 0.3]
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

# Предобработка текста
TRANSLATION_MAX_LENGTH = 512
GENERATE_RESPONSE_MAX_LENGTH = 1024

# Параметры оборудования
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
NUM_WORKERS = os.cpu_count()

# Вывод информации
print(f"Using device: {DEVICE}")
print(f"Number of workers: {NUM_WORKERS}")

-------------------------------------------------------------------------------------------------------------------------------------------------
import os
from pathlib import Path
import torch

# Directories
ROOT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = ROOT_DIR / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"

# Data files
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

# Machine learning models
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"
BACK_TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
ANSWER_MODEL_1_PATH = str(TRAINED_MODELS_DIR / "answer_model_1")
ANSWER_MODEL_2_PATH = str(TRAINED_MODELS_DIR / "answer_model_2")
ANSWER_MODEL_3_PATH = str(TRAINED_MODELS_DIR / "answer_model_3")
ENSEMBLE_WEIGHTS = [0.4, 0.3, 0.3]

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

# Text processing parameters
TRANSLATION_MAX_LENGTH = 512
GENERATE_RESPONSE_MAX_LENGTH = 1024

# Hardware parameters
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
NUM_WORKERS = os.cpu_count()

# Print information
print(f"Using device: {DEVICE}")
print(f"Number of workers: {NUM_WORKERS}")
import os
from pathlib import Path
import torch

# Directories
ROOT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = ROOT_DIR / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"

# Data files
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

# Machine learning models
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"
BACK_TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
ANSWER_MODEL_1_PATH = str(TRAINED_MODELS_DIR / "answer_model_1")
ANSWER_MODEL_2_PATH = str(TRAINED_MODELS_DIR / "answer_model_2")
ANSWER_MODEL_3_PATH = str(TRAINED_MODELS_DIR / "answer_model_3")
ENSEMBLE_WEIGHTS = [0.4, 0.3, 0.3]

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

# Text processing parameters
TRANSLATION_MAX_LENGTH = 512
GENERATE_RESPONSE_MAX_LENGTH = 1024

# Hardware parameters
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
NUM_WORKERS = os.cpu_count()

# Print information
print(f"Using device: {DEVICE}")
print(f"Number of workers: {NUM_WORKERS}")
'''