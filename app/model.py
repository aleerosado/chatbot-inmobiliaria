#simular rpta de modelo entrenado (se usara hasta antes de entrenar llaMA 3)
#def generate_response(prompt):
#    return f"[Respuesta simulada del modelo]: {prompt}"

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

# Ruta del modelo fine-tuneado
MODEL_PATH = "models/llama3_finetuned"

# Cargar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_response(prompt):
    # Preparar inputs y mover solo los tensores a la misma ubicación del modelo
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
