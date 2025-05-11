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
    torch_dtype=torch.float32,          # Usa float32 para evitar errores en CPU
    device_map="auto",                  # Distribuye entre CPU y GPU
    offload_folder="offload"            # Usa offload si es necesario
)

# Función para generar respuesta
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print("🧠 Generando respuesta...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_k=30,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("✅ Respuesta:")
    return response

# Modo interactivo
if __name__ == "__main__":
    print("\n🟢 Chatbot Inmobiliario Iniciado\n(Escribe 'salir' para terminar)\n")
    while True:
        prompt = input("🧑 Tú: ")
        if prompt.lower() in ["salir", "exit", "quit"]:
            print("👋 ¡Hasta luego!")
            break
        respuesta = generate_response(prompt)
        print(f"🤖 Bot: {respuesta}\n")
