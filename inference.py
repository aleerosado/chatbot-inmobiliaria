from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

# Ruta del modelo entrenado
MODEL_PATH = "models/llama3_finetuned"

# Cargar tokenizer y modelo
print("🔄 Cargando modelo y tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,      # usa float32 si estás offloadeando
    device_map="auto",
    offload_folder="offload"
)

# Función para generar respuesta
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
            do_sample=True,         # 🔥 modo creativo activado
            temperature=0.9,
            top_p=0.6,
            top_k=50
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Opcional: recorta el prompt de la respuesta
    respuesta_final = response.split("<|assistant|>")[-1].strip()
    return respuesta_final

# Modo interactivo
if __name__ == "__main__":
    print("\n🟢 Chatbot Inmobiliario Iniciado\n(Escribe 'salir' para terminar)\n")
    while True:
        user_input = input("🧑 Tú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("👋 ¡Hasta luego!")
            break
        respuesta = generate_response(user_input)
        print(f"🤖 Bot: {respuesta}\n")
