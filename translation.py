import torch
from transformers import MarianTokenizer, MarianMTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_translation_models():
    translation_model_name = "Helsinki-NLP/opus-mt-ru-en"
    back_translation_model_name = "Helsinki-NLP/opus-mt-en-ru"

    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name, src_lang="ru", tgt_lang="en")
    translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)
    translation_tokenizer.pad_token = translation_tokenizer.eos_token

    back_translation_tokenizer = MarianTokenizer.from_pretrained(back_translation_model_name, src_lang="en", tgt_lang="ru")
    back_translation_model = MarianMTModel.from_pretrained(back_translation_model_name).to(device)
    back_translation_tokenizer.pad_token = back_translation_tokenizer.eos_token

    return translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer

translation_model, translation_tokenizer, back_translation_model, back_translation_tokenizer = load_translation_models()

# Выводим список моделей в одном вызове функции print()
print(
    "Выберите модель:\n"
    "1. bert-base-uncased\n"
    "2. roberta-base\n"
    "3. gpt2\n"
    "4. rubert_cased_L-12_H-768_A-12_v2\n"
    "5. transfo-xl-wt103"
)

# В функции translate_text() включите аугментацию данных и fine-tuning для улучшения обработки естественного языка
def translate_text(model, tokenizer, text, target_language, src_language, max_length=512):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        outputs = model.generate(**inputs, max_length=1024, do_sample=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Произошла ошибка при переводе текста: {e}")
        translated_text = ""

    return translated_text

# Используйте функцию ensemble_predictions() для объединения предсказаний от разных моделей с учетом весов