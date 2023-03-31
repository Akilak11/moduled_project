import os

# Пути к данным
TRAIN_DATA_PATH = os.path.join("data", "train.csv")  # Путь к файлу с обучающими данными
TEST_DATA_PATH = os.path.join("data", "test.csv")  # Путь к файлу с тестовыми данными

# Пути к моделям
main_model_path = "mnt/c/Python_project/models/pretrain_models"  # Путь к главной модели
translation_model_path = "mnt/c/Python_project/models/pretrain_models/wmt19-en-ru"  # Путь к модели машинного перевода с английского на русский
back_translation_model_path = "mnt/c/Python_project/models/pretrain_models/wmt19-ru-en"  # Путь к модели машинного перевода с русского на английский

'''answer_models_paths = [
    str(MODELS_DIR / "answer_model_1"),
    str(MODELS_DIR / "answer_model_2"),
    str(MODELS_DIR / "answer_model_3"),
]

weights = [0.4, 0.3, 0.3]
#Веса ансамбля моделей
'''
# Параметры модели
BATCH_SIZE = 32  # Размер батча для обучения модели
EPOCHS = 10  # Количество эпох обучения
LEARNING_RATE = 0.001  # Скорость обучения модели
NUM_CLASSES = 10  # Количество классов для классификации
INPUT_SHAPE = (28, 28, 1)  # Размер входного изображения для модели

# Путь к моделям
MODELS_PATH = "models"  # Путь к директории с моделями

'''
ROOT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = ROOT_DIR / "models"

translation_model_path = str(MODELS_DIR / "opus-mt-ru-en")
back_translation_model_path = str(MODELS_DIR / "opus-mt-en-ru")
answer_models_paths = [
    str(MODELS_DIR / "answer_model_1"),
    str(MODELS_DIR / "answer_model_2"),
    str(MODELS_DIR / "answer_model_3"),
]

weights = [0.4, 0.3, 0.3]'''