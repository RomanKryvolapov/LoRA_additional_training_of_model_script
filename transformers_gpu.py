import os
import json
import torch
import shutil
import subprocess
import safetensors.torch
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

MODEL_DIR = Path("./model")
LORA_OUTPUT = Path("./lora")
MERGED_DIR = Path("./merged")
MERGED_CLEAN_DIR = Path("./merged_clean")
GGUF_DIR = Path("./gguf")
GGUF_PATH = GGUF_DIR / "model.gguf"
GGUF_CONVERTER = Path(r"C:\ExampleProjects\llama.cpp\convert_hf_to_gguf.py")
PYTHON_BIN = Path(".venv/Scripts/python.exe").resolve()

DATA_PATH = Path("gemma_lora_data.json")
MAX_SEQ_LEN = 4096
NUM_EPOCHS = 3
BATCH_SIZE = 2
LR = 2e-4

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


def main() -> None:
    print("[1/7] Cleaning previous artefacts…")
    for _dir in (LORA_OUTPUT, MERGED_DIR, GGUF_DIR):
        if _dir.exists():
            shutil.rmtree(_dir)
            print(f"  ‑ removed «{_dir}»")
    print("Finished cleaning")

    print("[2/7] Reading training examples…")
    with DATA_PATH.open(encoding="utf‑8") as fp:
        records = json.load(fp)

    def build_chat(example: dict) -> dict:
        prompt = example["prompt"].strip()
        response = example["response"].strip()
        return {
            "text": (
                f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n{response}<end_of_turn>\n"
            )
        }

    dataset = Dataset.from_list([build_chat(r) for r in records])
    print(f"Loaded {len(dataset):,} samples")

    print(f"[3/7] Loading base model from «{MODEL_DIR}» …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_quant_storage=torch.uint8
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    print("Model ready for fine‑tuning")

    print("[4/7] Starting supervised fine‑tuning …")
    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        return tokens
    ds_tok = dataset.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        data_collator=collator,
    )
    trainer.train()
    print("Training complete")

    print(f"[5/7] Saving LoRA weights to «{LORA_OUTPUT}» …")
    LORA_OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(LORA_OUTPUT),
        save_method="lora"
    )
    print("Adapters saved")

    print("[6/7] Merging LoRA + base ⟶ fp16 …")
    model = model.merge_and_unload()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        MERGED_DIR,
        safe_serialization=True,
        save_method="merged_16bit"
    )
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Merged model written to «{MERGED_DIR}»")

    print("[6.5/7] Creating cleaned model in «merged_clean» …")
    MERGED_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for file in MERGED_DIR.iterdir():
        if file.name != "model.safetensors":
            shutil.copy(file, MERGED_CLEAN_DIR / file.name)
    model_path = MERGED_DIR / "model.safetensors"
    clean_path = MERGED_CLEAN_DIR / "model.safetensors"
    state_dict = safetensors.torch.load_file(str(model_path))
    import re
    pattern = re.compile(
        r".*\.(absmax|zeros|scales|quant_map|quant_state(\..+)?|nested_absmax|nested_zeros|nested_scales|nested_quant_map)$"
    )
    keys_to_remove = [k for k in state_dict if pattern.match(k)]
    for key in keys_to_remove:
        del state_dict[key]
    safetensors.torch.save_file(state_dict, str(clean_path), metadata={"format": "pt"})
    print(f"Saved cleaned model to «{MERGED_CLEAN_DIR}», removed {len(keys_to_remove)} keys")

    print("[7/7] Converting to GGUF (q8_0) …")
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(PYTHON_BIN),
            str(GGUF_CONVERTER),
            os.path.abspath(MERGED_CLEAN_DIR),
            "--outfile", str(GGUF_PATH),
            "--outtype", "q8_0",
        ],
        check=True,
    )
    print(f"GGUF ready at «{GGUF_PATH}»")
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
