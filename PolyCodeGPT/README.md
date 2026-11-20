# PolyCodeGPT

## Project Introduction

PolyCodeGPT is a small, multilingual, programming-oriented language model built **from scratch** using PyTorch. The model is trained from random initialization with no pretrained weights. It targets programming content (initially Python and JavaScript) alongside technical text in English, French, and Arabic.

## Project Objectives

- Train a byte-level BPE tokenizer on a project-specific corpus.
- Build a custom decoder-only Transformer architecture.
- Use random parameter initialization (no pretrained LLM weights).
- Implement autoregressive next-token prediction.
- Support programming content: Python, JavaScript, and additional languages.
- Support multilingual technical text: English, French, and Arabic.
- Provide pretraining, evaluation, and optional instruction tuning.
- Offer an optional API and chatbot interface.

## Main Technical Characteristics

- **Framework:** PyTorch 2.15.0+cu132
- **CUDA:** 13.2 (supports Blackwell / sm_120)
- **Tokenizer:** Byte-level BPE (trained from scratch)
- **Architecture:** Decoder-only Transformer
- **Initialization:** Random (no pretrained weights)
- **Training:** Autoregressive next-token prediction

## Proposed Architecture

A decoder-only Transformer with:
- Byte-level BPE tokenization
- Multi-head self-attention
- Feed-forward networks with GELU activation
- Layer normalization
- Residual connections
- Tied input/output embeddings

## Initial Model Configuration

| Parameter | Value |
|-----------|-------|
| Vocab size | 16,000 |
| Model dimension (d_model) | 384 |
| Number of layers | 6 |
| Number of heads | 6 |
| FFN dimension | 1,536 |
| Context length | 512 |
| Dropout | 0.1 |
| Tie embeddings | Yes |

These are initial experimental values subject to adjustment.

## Project Directory Structure

```
PolyCodeGPT/
├── configs/          # YAML configuration files
├── data/             # Raw, interim, processed data and manifests
├── tokenizer/        # Tokenizer artifacts
├── checkpoints/      # Model checkpoints
├── logs/             # Training logs
├── reports/          # Evaluation reports
├── notebooks/        # Jupyter notebooks
├── scripts/          # Utility scripts
├── docs/             # Documentation
├── tests/            # Unit and integration tests
├── src/polycodegpt/  # Source code package
├── .gitignore
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Development Roadmap

1. **Phase 0** — Environment audit ✅ Completed
2. **Phase 1** — Project setup and environment verification 🔄 In Progress
3. **Phase 2** — Data collection and preprocessing
4. **Phase 3** — Tokenizer training
5. **Phase 4** — Model architecture implementation
6. **Phase 5** — Pretraining
7. **Phase 6** — Evaluation
8. **Phase 7** — Optional instruction tuning and chatbot interface

## Environment Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support (tested on RTX 5060 Laptop GPU)
- CUDA 13.2 compatible PyTorch

### Setup Steps

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt

# Verify PyTorch CUDA installation
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
```

### CUDA Environment Verification

```bash
.venv\Scripts\python.exe scripts/check_environment.py
```

## Current Development Status

- **Phase 0:** Completed
- **Phase 1:** In Progress
- **Phase 2 onward:** Not Started

> PolyCodeGPT is being developed from random initialization. No pretrained LLM weights are used. Instruction tuning and the chatbot interface are optional extensions.
