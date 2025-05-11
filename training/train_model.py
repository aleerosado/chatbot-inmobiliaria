import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuración
MODEL_ID = "NousResearch/Llama-2-7b-hf"
CSV_PATH = "training/dataset.csv"
OUTPUT_DIR = "models/test_finetune"

# Cargar dataset
df = pd.read_csv(CSV_PATH)

# 🔽 AQUI VA LA FUNCIÓN
def format_row(example):
    return f"<|system|>\nEres un asesor inmobiliario profesional.\n<|user|>\n{example['instruction']}: {example['input']}\n<|assistant|>\n{example['output']}"

# Aplicar la función
df_formatted = df.apply(format_row, axis=1).tolist()

# Crear dataset compatible con HuggingFace
dataset = Dataset.from_dict({"text": df_formatted})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True)

# Modelo base en 4bit
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)
model = prepare_model_for_kbit_training(model)

# Configurar LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Argumentos de entrenamiento mínimos
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    max_steps=4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none"
)

# Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Guardar modelo
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Mini modelo entrenado y guardado en: {OUTPUT_DIR}")
