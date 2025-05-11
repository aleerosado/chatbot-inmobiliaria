from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Rutas
PEFT_MODEL_PATH = "models/test_finetune"
MERGED_MODEL_PATH = "models/test_finetune_merged"

# Cargar config PEFT
peft_config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)

# Cargar modelo base
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# Cargar adaptadores
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)

# Fusionar adaptadores LoRA al modelo base
model = model.merge_and_unload()

# Guardar modelo fusionado
model.save_pretrained(MERGED_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print(f"✅ Modelo fusionado guardado en: {MERGED_MODEL_PATH}")
