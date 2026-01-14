# LLaMA-2 QLoRA Fine-Tuning

This repository contains a minimal and reproducible setup for fine-tuning **LLaMA-2-7B-Chat** using **QLoRA (4-bit quantization)** and **Supervised Fine-Tuning (SFT)** with the `trl` library.

The project is intentionally kept minimal, with a clear separation between training and inference, while preserving the original fine-tuning code logic.

---

## Overview

* **Base Model:** NousResearch/Llama-2-7b-chat-hf
* **Fine-Tuning Method:** QLoRA (NF4, 4-bit)
* **Trainer:** TRL `SFTTrainer`
* **Dataset:** `mlabonne/guanaco-llama2-1k`
* **Task:** Causal Language Modeling (Chat format)

---

## Repository Structure

```
.
├── requirements.txt
├── train.py
└── inference.py
```

* `train.py` — Fine-tunes the base LLaMA-2 model using QLoRA and saves the trained adapters.
* `inference.py` — Loads the fine-tuned model and runs text generation.
* `requirements.txt` — Lists all required dependencies.

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have access to the LLaMA-2 model on Hugging Face and are logged in:

```bash
huggingface-cli login
```

---

## Training

Run the fine-tuning script:

```bash
python train.py
```

The fine-tuned model and tokenizer will be saved to:

```
llama-2-7b-chat-guanaco/
```

---

## Inference

After training, run inference using:

```bash
python inference.py
```

You can modify the prompt directly inside `inference.py` to test different inputs.

---

## Notes

* This setup uses 4-bit quantization to enable fine-tuning large language models on limited GPU memory.
* The configuration is intentionally minimal and suitable for experimentation or educational purposes.
* The code can be extended to support multi-GPU training (DDP/FSDP) or adapter merging if required.

