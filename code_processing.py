# Функция separate_code_and_explanations принимает сгенерированный код и возвращает два списка: список строк с кодом и список строк с объяснениями, начинающихся с символа #.
def separate_code_and_explanations(generated_code):
    lines = generated_code.split("\n")
    code_lines = [line for line in lines if not line.startswith("#")]
    explanations = [line for line in lines if line.startswith("#")]
    return code_lines, explanations

#Функция combine_code_and_translated_explanations принимает список строк с кодом и список строк с переведенными объяснениями и возвращает строку, которая объединяет эти списки вместе.
#Для этого она итерируется по строкам кода и, если строка начинается с символа #, заменяет ее на соответствующее переведенное объяснение из списка translated_explanations.
def combine_code_and_translated_explanations(code_lines, translated_explanations):
    lines = code_lines + translated_explanations
    final_result = "\n".join([code_line if not code_line.startswith("#") else translated_explanations.pop(0) for code_line in lines])
    return final_result
