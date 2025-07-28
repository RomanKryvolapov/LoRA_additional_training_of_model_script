import os
import shutil
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
from trl import SFTTrainer

START_TAG = "<start_of_turn>"
END_TAG = "<end_of_turn>"
USER_ROLE_TAG = "user"
MODEL_ROLE_TAG = "model"

BASE_MODEL_PATH = "./model"
LORA_DIR = "./lora"
MERGED_DIR = "./merged"
GGUF_PATH = os.path.abspath("gguf/model.gguf")
CONVERTER_SCRIPT = "C:\ExampleProjects\llama.cpp\convert_hf_to_gguf.py"
PYTHON_EXE = os.path.abspath(".venv/Scripts/python.exe")

def to_chat(prompt: str, response: str, bos: str = "") -> str:
    return (
        f"{bos}{START_TAG}{USER_ROLE_TAG}\n{prompt}{END_TAG}\n"
        f"{START_TAG}{MODEL_ROLE_TAG}\n{response}{END_TAG}"
    )

print("Cleaning output directories...")
for path in (LORA_DIR, MERGED_DIR, os.path.dirname(GGUF_PATH)):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, device_map="auto")

print("Loading dataset gemma_lora_data.json...")
raw_ds = load_dataset("json", data_files="gemma_lora_data.json")["train"]
print(f"   → Loaded {len(raw_ds):,} examples with columns {raw_ds.column_names}")

def fmt(example):
    return {
        "text": to_chat(
            example["prompt"],
            example["response"],
            tokenizer.bos_token or ""
        )
    }

dataset = raw_ds.map(fmt, remove_columns=raw_ds.column_names)
print("Dataset converted to chat‑template.")

print("Applying LoRA configuration...")
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)

print("Starting LoRA fine‑tuning...")
train_args = TrainingArguments(
    output_dir=LORA_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=train_args,
    dataset_text_field="text",
)
trainer.train()

print("Saving LoRA adapter...")
model.save_pretrained(LORA_DIR)

print("Merging LoRA adapter with base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, torch_dtype=torch.float16
)
base_model = PeftModel.from_pretrained(base_model, LORA_DIR)
base_model = base_model.merge_and_unload()
base_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print("Merged model saved to", MERGED_DIR)

print("Converting merged model to GGUF format...")
subprocess.run(
    [
        PYTHON_EXE,
        CONVERTER_SCRIPT,
        os.path.abspath(MERGED_DIR),
        "--outfile",
        GGUF_PATH,
        "--outtype",
        "f16",
    ],
    check=True,
)
print(f"GGUF model saved to: {GGUF_PATH}")