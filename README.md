# ChemLLM-MT (Version 1)

[Russian Version Below]

**ChemLLM-MT** is a lightweight Transformer-based (Seq2Seq) model designed to predict organic chemical reactions and retrosynthetic pathways. It was trained from scratch using the **Intel Arc B580 (XPU)** hardware, proving the viability of non-NVIDIA accelerators for deep learning tasks.

## Key Features
- **Multi-task Learning:** One model handles both direct synthesis (`[SYNTHESIS]`) and retrosynthesis (`[RETRO]`) using task prompts.
- **SMILES Representation:** Processes chemical structures as simplified molecular-input line-entry system strings.
- **XPU/CUDA/CPU Support:** Automatically detects and uses Intel GPU, NVIDIA GPU, or CPU.
- **RDKit Integration:** Includes automated chemical validity checks and Tanimoto similarity metrics.

## Model Architecture
- **Type:** Transformer Encoder-Decoder (Seq2Seq)
- **Parameters:** ~25.4 Million
- **Layers:** 6 (Encoder) / 6 (Decoder)
- **Embedding Dim:** 384
- **Normalization:** Pre-LayerNorm (for training stability)

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd llm-test
   ```

2. **Setup environment:**
   ```bash
   uv venv
   uv sync
   ```

3. **Prepare Data:**
   Ensure `datset.csv` is in the root, then convert it to Parquet for speed:
   ```bash
   python convert_to_parquet.py
   ```

---

## Usage

### Training
To train the model (or resume from last checkpoint):
```bash
python train.py
```

### Prediction (Inference)
Run the interactive prediction utility. You can select a specific epoch to test:
```bash
python predict.py
```

### Benchmarking
Run an automated test suite across all saved checkpoints:
```bash
python benchmark.py
```

### Visualization
Generate training loss and learning rate graphs:
```bash
python plot_metrics.py
python plot_benchmarks.py
```

---

## Model Card Summary
- **Dataset:** ~50k chemical reactions.
- **Final Loss:** 0.1468 (approx. 86% token confidence).
- **Chemical Logic:** The model excels at recognizing functional groups (Boc, Nitro) but may exhibit "structural over-generation" in early stages.

---
