#simular rpta de modelo entrenado (se usara hasta antes de entrenar llaMA 3)
#def generate_response(prompt):
#    return f"[Respuesta simulada del modelo]: {prompt}"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#modelo entrenado
MODEL_PATH = "models/llama3_finetuned"

# Cargar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
