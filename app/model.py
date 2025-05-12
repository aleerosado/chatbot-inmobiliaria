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
    torch_dtype=torch.float32,  # Usa float16 si tienes GPU potente
    device_map="auto"
)

# 🔁 Función de generación de texto
def generate_response(user_input):
    try:
        print(f"🧪 [MODEL] Prompt recibido:\n{user_input}")

        prompt = f"""<|system|>
Eres un asesor inmobiliario profesional.
<|user|>
{user_input}
<|assistant|>
"""

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print(f"🧠 Generando respuesta...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,        
                do_sample=False,          
                temperature=0.7,
                top_p=1.0,
                top_k=0
            )
        print("✅ Generación completada")


        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_response = decoded.split("<|assistant|>")[-1].strip()
        print(f"✅ [MODEL] Respuesta generada:\n{final_response}")
        return final_response

    except Exception as e:
        print("❌ [MODEL] Error al generar respuesta:", str(e))
        return "Lo siento, ocurrió un error generando la respuesta."
