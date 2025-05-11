import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "TheBloke/Llama-2-7B-Chat-GPTQ"
CSV_PATH = "training/dataset.csv"
OUTPUT_DIR = "models/llama3_finetuned"

#cargando dataset csv
df = pd.read_csv(CSV_PATH)

#unir campos para el finetuning 
def format_row(example):
    return f"<|system|>\nEres un asesor inmobiliario profesional.\n<|user|>\n{example['instruction']}: {example['input']}\n<|assistant|>\n{example['output']}"

df_formatted = df.apply(format_row, axis=1).tolist()
dataset = Dataset.from_dict({"text": df_formatted})

#tokenizando : convirtiendo en texto para llama3
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

#cargando modelo base y preparando para lora(adaptacion del modelo de aprendizaje para que no se vuelva a entrenar todo el modelo)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

#Entrenamiento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    fp16=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()

#guardando el modelo luego del fine tuning
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Modelo entrenado y guardado en: {OUTPUT_DIR}")