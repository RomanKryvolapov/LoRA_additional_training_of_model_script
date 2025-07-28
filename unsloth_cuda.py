# ------------------------ Model in folder = google/gemma-3-4b-it ---------------------------
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
import os
os.environ["UNSLOTH_PATCH_RL_TRAINERS"] = "false"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
import json
import shutil
import subprocess
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

MODEL_DIR = Path("./model")
LORA_OUTPUT = Path("./lora")
MERGED_DIR = Path("./merged")
GGUF_DIR = Path("./gguf")
GGUF_PATH = GGUF_DIR / "model.gguf"
GGUF_CONVERTER = Path(r"C:\ExampleProjects\llama.cpp\convert_hf_to_gguf.py")
PYTHON_BIN = Path(".venv/Scripts/python.exe").resolve()
DATA_PATH = Path("gemma_lora_data.json")
MAX_SEQ_LEN = 4096
NUM_EPOCHS = 3
BATCH_SIZE = 2
LR = 2e-4

def main():
    print("[1/7] Cleaning previous artefacts…")
    for _dir in (LORA_OUTPUT, MERGED_DIR, GGUF_DIR):
        if _dir.exists():
            shutil.rmtree(_dir)
            print(f"  ‑ removed «{_dir}»")
    print("Finished cleaning")

    print("[2/7] Reading training examples…")
    with DATA_PATH.open(encoding="utf-8") as fp:
        records = json.load(fp)
    def build_chat(example):
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
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_DIR),
    )
    model = FastLanguageModel.get_peft_model(model)
    print("Model ready for fine‑tuning")

    print("[4/7] Starting supervised fine‑tuning …")
    sft_cfg = SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_cfg,
    )
    trainer.train()
    print("Training complete")

    print(f"[5/7] Saving LoRA weights to «{LORA_OUTPUT}» …")
    LORA_OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(LORA_OUTPUT),
        tokenizer,
        save_method="lora"
    )
    print("Adapters saved")

    print("[6/7] Merging LoRA + base ⟶ fp16 …")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(MERGED_DIR),
        tokenizer,
        safe_serialization=True,
        save_method="merged_16bit"
    )
    print(f"Merged model written to «{MERGED_DIR}»")

    print("[7/7] Converting to GGUF (q8_0) …")
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(PYTHON_BIN),
            str(GGUF_CONVERTER),
            os.path.abspath(MERGED_DIR),
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
