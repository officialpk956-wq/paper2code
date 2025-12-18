# paper2code

**paper2code** is a research-oriented project that aims to convert deep learning research paper architectures into structured, executable code and clear architecture diagrams.  
It helps bridge the gap between **research papers** and **practical implementation** by making model architectures explicit, reproducible, and verifiable.

---

## Why paper2code?

Reproducing deep learning papers is often difficult because:
- architectures are described informally
- implementation details are missing or ambiguous
- diagrams are incomplete or inconsistent

paper2code addresses these issues by converting paper-level descriptions into:
- structured schemas
- executable model representations
- clear architecture diagrams

---

## Key Features

- Architecture extraction from research papers
- Modular support for multiple model families:
  - ResNet
  - U-Net
  - Vision Transformer (ViT)
  - Transformer (Encoder–Decoder)
- Automatic architecture diagram generation
- Code-ready schema generation
- Parameter counting and FLOPs estimation
- Model verification utilities
- Clean and extensible project design

---

## Project Structure

```text
paper2code/
├── main.py                  # Entry point
├── src/                     # Core implementation
│   ├── architecture_extractor.py
│   ├── blocks_*.py           # Model building blocks
│   ├── diagram_*.py          # Diagram generation
│   ├── schema_*.py           # Schema definitions
│   ├── schema_refiner_*.py   # Architecture-specific rules
│   ├── model_builder.py
│   ├── param_counter.py
│   ├── flops_estimator.py
│   ├── verify_model.py
│   └── run_*_codegen.py
├── paper2code/               # Internal package
├── requirements.txt
└── README.md


## About Data and Generated Files

This repository intentionally does **not** include datasets, trained models, or generated outputs such as diagrams and schemas.

Folders like `data/`, `outputs/`, `models/`, and experiment artifacts are **excluded** to:
- keep the repository lightweight
- avoid large or copyrighted files
- encourage reproducibility through code

All diagrams, schemas, and model representations can be regenerated locally using the provided scripts.

