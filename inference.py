from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Ruta al modelo fine-tuneado
MODEL_PATH = "models/test_finetune"

print("🔄 Cargando modelo y tokenizer...")

# 1. Cargar configuración PEFT
peft_config = PeftConfig.from_pretrained(MODEL_PATH)

# 2. Cargar modelo base original
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float32,  # usa torch.float16 si estás en GPU potente
    device_map="auto"
)

# 3. Cargar adaptadores LoRA encima del modelo base
model = PeftModel.from_pretrained(base_model, MODEL_PATH)

# 4. Cargar tokenizer entrenado
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 🧠 Función para generar respuesta
def generate_response(user_input):
    prompt = f"""<|system|>
Eres un asesor inmobiliario profesional.
<|user|>
{user_input}
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("🧠 Generando respuesta...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.9,
            top_p=0.6,
            top_k=50
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    respuesta_final = response.split("<|assistant|>")[-1].strip()
    return respuesta_final

# 💬 Modo interactivo por consola
if __name__ == "__main__":
    print("\n🟢 Chatbot Inmobiliario Iniciado\n(Escribe 'salir' para terminar)\n")
    while True:
        user_input = input("🧑 Tú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("👋 ¡Hasta luego!")
            break
        respuesta = generate_response(user_input)
        print(f"🤖 Bot: {respuesta}\n")
