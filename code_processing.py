def separate_code_and_explanations(generated_code):
    lines = generated_code.split("\n")
    code_lines = [line for line in lines if not line.startswith("#")]
    explanations = [line for line in lines if line.startswith("#")]
    return code_lines, explanations

def combine_code_and_translated_explanations(code_lines, translated_explanations):
    lines = code_lines + translated_explanations
    final_result = "\n".join([code_line if not code_line.startswith("#") else translated_explanations.pop(0) for code_line in lines])
    return final_result
