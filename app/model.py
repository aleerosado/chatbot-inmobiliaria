#simular rpta de modelo entrenado (se usara hasta antes de entrenar llaMA 3)
#def generate_response(prompt):
#    return f"[Respuesta simulada del modelo]: {prompt}"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Ruta al modelo fine-tuneado y fusionado
MODEL_PATH = "models/test_finetune_merged"

# ✅ Cargar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,   # si tienes GPU con más RAM, puedes usar float16
    device_map="auto"
)

# 🔁 Función de generación de texto
def generate_response(user_input):
    prompt = f"""<|system|>
Eres un asesor inmobiliario profesional.
<|user|>
{user_input}
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.9,
            top_p=0.6,
            top_k=50
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()

