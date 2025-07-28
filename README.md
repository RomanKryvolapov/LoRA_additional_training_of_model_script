* Here are 2 scripts that allow you to further train LLM models using LoRA.
* The model must be downloaded in advance.
* I did this because of unstable authorization on huggingface.
* The scripts use the markup for Gemma, you can modify it for any other model https://huggingface.co/google/gemma-3-4b-it
* unsloth_cuda.py uses https://github.com/unslothai/unsloth
* You need to use your own path for the file convert_hf_to_gguf.py
* Download the repository https://github.com/ggml-org/llama.cpp it contains the script for conversion
* The gguf model can be run in https://lmstudio.ai/
