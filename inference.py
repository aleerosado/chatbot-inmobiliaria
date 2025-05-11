from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

# Ruta al modelo entrenado
MODEL_PATH = "models/test_finetune"

# Cargar tokenizer y modelo
print("🔄 Cargando modelo y tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,       # usa float16 si tu GPU lo permite
    device_map="auto",
    offload_folder="offload",
    is_trainable=False,
    low_cpu_mem_usage=False,
    mean_resizing=False
)

# Generador de respuesta
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
