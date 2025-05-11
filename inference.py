from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ Ruta al modelo fusionado (base + LoRA integrados)
MODEL_PATH = "models/test_finetune_merged"

print("🔄 Cargando modelo y tokenizer...")

# Cargar tokenizer entrenado
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Cargar modelo entrenado y fusionado
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="auto"
)

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

# Modo interactivo CLI
if __name__ == "__main__":
    print("\n🟢 Chatbot Inmobiliario Iniciado\n(Escribe 'salir' para terminar)\n")
    while True:
        user_input = input("🧑 Tú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("👋 ¡Hasta luego!")
            break
        respuesta = generate_response(user_input)
        print(f"🤖 Bot: {respuesta}\n")
