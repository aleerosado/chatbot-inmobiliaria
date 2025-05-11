import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

# Configuración general
MODEL_ID = "NousResearch/Llama-2-7b-hf"
CSV_PATH = "training/dataset.csv"
OUTPUT_DIR = "models/test_finetune"

# Cargar dataset desde CSV
df = pd.read_csv(CSV_PATH)

# Formatear cada fila en formato de instrucción
def format_row(example):
    return f"<|system|>\nEres un asesor inmobiliario profesional.\n<|user|>\n{example['instruction']}: {example['input']}\n<|assistant|>\n{example['output']}"

df_formatted = df.apply(format_row, axis=1).tolist()
dataset = Dataset.from_dict({"text": df_formatted})

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Añadir tokens especiales si no están
special_tokens = {
    "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]
}
tokenizer.add_special_tokens(special_tokens)

# Tokenizar datos
def tokenize(batch):
    tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True)

# Cargar modelo base con soporte 4-bit
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

# Redimensionar embeddings para incluir tokens especiales
model.resize_token_embeddings(len(tokenizer))

# Argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=2,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy="epoch",
    report_to="none"
)

# Crear entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Entrenar modelo
trainer.train()

# Guardar modelo y tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Modelo fine-tuneado y guardado en: {OUTPUT_DIR}")
