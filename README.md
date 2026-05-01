# Resilient LLM Fine-Tuning & Inference Framework

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9D423?logo=huggingface)
RichUI

An end-to-end, fault-tolerant pipeline engineered for fine-tuning and running inference on Large Language Models (up to 14B parameters) in distributed and highly constrained cloud environments (like Google Colab/Shared GPUs). 

This repository demonstrates advanced **MLOps**, **Distributed Systems Engineering**, and **Memory Optimization** techniques, ensuring that long-running ML jobs survive session timeouts, network drops, and Out-Of-Memory (OOM) errors.

---

## Key Engineering Features

### 1. Fault-Tolerant Checkpointing & Atomic State Recovery
Cloud environments are preemptible. This framework guarantees zero data loss:
* **Atomic Operations:** Uses `.tmp` file writing and `os.replace()` to ensure checkpoints and output CSVs are never corrupted during a crash.
* **Emergency Save Handlers:** Integrates `atexit` and `signal.signal(SIGTERM)` to catch instance kills and forcefully flush the `batch_buffer` to disk before the VM dies.
* **Seamless Resumption:** Deep checkpoint parsing allows the inference and training loops to pick up exactly at the last processed tweet/batch.

### 2. Dynamic VRAM Management & "Smart OOM Catcher"
Handling a 14B model on a single T4 GPU requires extreme memory efficiency:
* **Adaptive Batch Sizing:** Computes upcoming token lengths against a `TOKEN_BUDGET` and current `vram_reserved_gb` to dynamically scale the batch size up or down *during* the loop.
* **Graceful Degradation:** If `torch.cuda.OutOfMemoryError` is caught, the pipeline automatically flushes memory, shrinks the batch size by 20%, and retries the exact same sequence without failing the pipeline.
* **4-bit NF4 Quantization:** Implements Double Quantization via `BitsAndBytes` to fit massive Qwen models into ~8GB of VRAM.

### 3. High-Performance SFT Pipeline
* **Knowledge Distillation:** Pipeline engineered to use outputs from larger Qwen 14B model to train smaller, efficient 4B/9B parameter student models.
* **Custom SFTTrainer:** Overrides base HuggingFace classes to implement hard-balanced category sampling, custom cross-entropy loss masking, and safe directory resolution across distributed drives.

### 4. Beautiful, Real-Time MLOps Telemetry
Built entirely with `richUI`, the pipeline provides a commercial-grade terminal UI:
* Real-time ETA, throughput (tweets/sec), and VRAM tracking.
* Live-streaming of the model's Chain-of-Thought (`<think>`) generation.
* Richly formatted post-run analysis panels (Confusion Matrices, Precision/Recall, Error Rates).

---

## 📸 Pipeline Visualizations

### The Inference Engine (Live Telemetry)
*The custom `rich` dashboard tracking VRAM usage, dynamic batch sizing, and real-time generation metrics.*
> **<img width="1012" height="315" alt="image" src="https://github.com/user-attachments/assets/f6922092-5b86-4d5e-b2eb-25785573dca5" />**

### Post-Classification Analytics
*Auto-generated confusion matrices and confidence statistics rendered directly in the terminal.*
> **[❗ ACTION: Insert a screenshot here of the "Block F-A Output" showing the "CATEGORY DISTRIBUTION", "Confidence Statistics", and "Label Correction Summary" panels (Page 38/39 of your PDF)]**

### Distributed Result generation and saving
*Matplotlib-generated artifacts tracking DEI classification shifts across different university datasets.*
> **<img width="993" height="417" alt="image" src="https://github.com/user-attachments/assets/4cc1f52e-a59f-4c92-86a7-04d446f9106e" />**

---

## ⚙️ Architecture Overview

```mermaid
graph TD;
    A[Data Ingestion & Stratification] --> B[Tokenizer & Prompt Formatting];
    B --> C[4-bit Base Model Loading];
    C --> D[QLoRA Adapter Injection];
    D --> E[Custom DEITrainer Loop];
    E --> F[Atomic Save & Evaluation];
    
    subgraph Inference Engine
    G[Smart OOM Catcher] --> H[Dynamic Batch Sizing];
    H --> I[TextIteratorStreamer];
    I --> J[Atomic Output Flush];
    end
