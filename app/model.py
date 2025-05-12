#simular rpta de modelo entrenado (se usara hasta antes de entrenar llaMA 3)
#def generate_response(prompt):
#    return f"[Respuesta simulada del modelo]: {prompt}"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Ruta al modelo fusionado
MODEL_PATH = "models/test_finetune_merged"

# Cargar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="auto"
)

def generate_response(user_input):
    try:
        print(f"🧪 [MODEL] Prompt recibido:\n{user_input}")

        inputs = tokenizer(user_input, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print("🧠 Generando respuesta...")
        start = time.time()

        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

        end = time.time()
        print(f"✅ Generación completada")
        print(f"⏱️ Tiempo de generación: {round(end - start, 2)}s")

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_response = decoded.split("<|assistant|>")[-1].strip()

        if len(final_response) > 300:
            final_response = final_response[:300] + "..."

        print(f"✅ [MODEL] Respuesta generada:\n{final_response}")
        return final_response

    except Exception as e:
        print("❌ [MODEL] Error al generar respuesta:", str(e))
        return "Lo siento, hubo un error generando la respuesta."
