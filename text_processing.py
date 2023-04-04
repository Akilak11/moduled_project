#модуль text_processing.py
import re
from code_processing import separate_code_and_explanations, combine_code_and_translated_explanations
from typing import Tuple

#Функция принимает строку текста и заменяет все символы, кроме букв, цифр и некоторых знаков препинания, на пробелы. Затем она удаляет лишние пробелы в тексте и возвращает очищенный текст.
def clean_text(text):
    cleaned_text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9\s\.,!?;:(){}\[\]<>+=\-/\\*%|&'\"_]", " ", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text

def extract_code_and_explanations(text: str) -> Tuple[str, str]:
    code_lines = []
    explanation_lines = []

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            explanation_lines.append(line)
        elif line:
            code_lines.append(line)

    code = "\n".join(code_lines)
    explanation = "\n".join(explanation_lines)

    return code, explanation

'''def translate_and_combine(text: str) -> str:
    # Замените "ru" и "en" на нужные языки, если нужно переводить на другие языки
    translated_text = separate_code_and_explanations(text)
    translated_explanations = translate_text(translated_text[1], target_language="en", src_language="ru")
    return combine_code_and_translated_explanations(translated_text[0], translated_explanations)'''

def validate_input(user_input, min_length=1, max_length=512):
    """
    Проверяет, соответствует ли ввод пользователя заданным критериям.
    
    Args:
        user_input (str): Введенный пользователем текст.
        min_length (int): Минимальная допустимая длина текста.
        max_length (int): Максимальная допустимая длина текста.
        
    Returns:
        bool: True, если ввод соответствует критериям, иначе False.
    """
    input_length = len(user_input)
    
    if input_length < min_length or input_length > max_length:
        return False
        
    # Добавьте дополнительные проверки здесь, если необходимо
    
    return True
