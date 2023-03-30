# Функция для очистки текста с использованием регулярных выражений
def clean_text(text):
    cleaned_text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9\s\.,!?;:(){}\[\]<>+=\-/\\*%|&'\"_]", " ", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text