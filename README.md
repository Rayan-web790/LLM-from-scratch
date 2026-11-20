# LLM-from-scratch (PolyCodeGPT)

A small, multilingual, programming-oriented language model built **from scratch** using PyTorch.

## Overview

PolyCodeGPT is trained from random initialization with no pretrained weights. It targets programming languages (Python and JavaScript) alongside technical text in English, French, and Arabic.

## Repository Structure

```text
LLM FROM SCRATCH/
├── PolyCodeGPT/               # Main project package and codebase
│   ├── configs/               # Model and training YAML configurations
│   ├── docs/                  # Design specs, licensing, and strategy documentation
│   ├── functional/            # Functional specifications and evaluation suites
│   ├── reports/               # Experiment logs and phase reports
│   ├── scripts/               # Pretraining, evaluation, and analysis scripts
│   ├── src/polycodegpt/       # Core library: architecture, dataset, engine, tokenization
│   ├── tests/                 # Unit and integration test suites
│   ├── tokenizer/             # Byte-level BPE tokenizer configs and metadata
│   ├── pyproject.toml         # Build system and project metadata
│   └── requirements.txt       # Python dependencies
├── reports/                   # Phase milestone reports
└── README.md
```

## Quick Start for Teammates

### 1. Clone the Repository
```bash
git clone https://github.com/Rayan-web790/LLM-from-scratch.git
cd LLM-from-scratch
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
cd PolyCodeGPT
pip install -r requirements.txt
pip install -e .
```

### 4. Run Tests
```bash
pytest
```
