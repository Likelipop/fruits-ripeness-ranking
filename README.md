# Fruit Ripeness Ranking: Transfer Learning & CPU Quantization Benchmark

This repository is a reproducible research and benchmarking sandbox designed to optimize **fruit ripeness classification models** using **Transfer Learning** and **Post-Training Static Quantization (PTSQ)** for CPU-bound inference.

---

## 🚀 Workflow Overview

The project uses a hybrid **Colab-to-Local** workflow:

```mermaid
graph TD
    A[0. Raw Dataset] --> B(1. Colab/GPU: Model Training)
    B -->|Save Weight Checkpoint| C[Google Drive]
    C -->|Download Checkpoint locally| D(2. Local Laptop: Quantization Calibration & CPU Inference Benchmark)
    D -->|Generates| E[quantization_report.md]
```

1. **Model Training (Google Colab / GPU)**:
   - Train a quantizable classification model (e.g. ResNet18) on your dataset using GPU acceleration.
   - Save the model weights checkpoint (e.g. `fruit_model.pth`) and download it to your local machine.
2. **Quantization & Benchmarking (Local / CPU)**:
   - Perform Post-Training Static Quantization (PTSQ) using local representative calibration images on CPU.
   - Benchmark model size, evaluation accuracy, and CPU inference latency between the float32 and int8 checkpoints.
   - Generate an automated performance report (`quantization_report.md`).

---

## 📁 Repository Structure

```
├── config/
│   └── config.yaml            # Optional pipeline configuration
├── data/
│   ├── Train/                 # Training set (used locally for quantization calibration)
│   │   ├── Overripe/
│   │   ├── Ripe/
│   │   └── Unripe/
│   └── Test/                  # Test set (used for CPU benchmarking)
│       ├── Overripe/
│       ├── Ripe/
│       └── Unripe/
├── models/                    # Directory where model weights are stored (ignored by git)
│   ├── fruit_model.pth        # Float32 model checkpoint downloaded from Colab
│   └── fruit_model_quantized.pth # Quantized INT8 model state dict (generated locally)
├── src/
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       └── quantize.py        # Model definition, dataloaders, PTSQ quantization, & CPU benchmarking
├── main.py                    # Entrypoint CLI for quantization and benchmarking
├── pyproject.toml             # Configuration & package dependencies
└── requirements.txt           # Standard pip requirements file
```

---

## 🛠️ Setup & Installation

### 1. Requirements
Ensure you have Python **>= 3.13** installed. The environment uses PyTorch, Torchvision, and utility libraries.

### 2. Installation using `uv` (Recommended)
This project is configured with `uv` for high-speed package management:
```bash
# Sync dependencies and activate virtual environment
uv sync
source .venv/bin/activate
```

### 3. Installation using standard `pip`
Alternatively, create a virtual environment and install standard dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🏃 Replication Guide

Follow these steps to replicate the experiment:

### Step 1: Place Your Model Checkpoint
Place your trained float32 model checkpoint (state dictionary saved from your Colab training run) in a `models/` directory:
```bash
mkdir -p models
# Move your downloaded weight file here:
# models/fruit_model.pth
```

### Step 2: Run the Quantization and Benchmarking Pipeline
To run static quantization, save the quantized model, and benchmark performance metrics on CPU:
```bash
python main.py --run-all models/fruit_model.pth
```
This single command will:
1. Load your Float32 model.
2. Calibrate activations on a representative subset of the training set.
3. Convert weights and activations to INT8 precision.
4. Save the quantized state dict to `models/fruit_model_quantized.pth`.
5. Benchmark size, accuracy, and latency, and write a summary table to `quantization_report.md`.

### Alternative CLI Command Commands

* **Quantize Only**:
  ```bash
  python main.py --quantize models/fruit_model.pth --out-quantized models/fruit_model_quantized.pth
  ```

* **Benchmark Only**:
  ```bash
  python main.py --benchmark models/fruit_model.pth models/fruit_model_quantized.pth
  ```

## 🔬 Quantization Theory & Learnings

### Motivation
I wanted to learn how quantization works under the hood and how it affects model footprint, inference latency, and accuracy when deploying to resource-constrained environments like commodity CPUs. 

### What I Learned
Through this project, I learned the following:
1. **Dynamic vs. Static Quantization**: I learned that dynamic quantization only quantizes weights offline (activations are quantized on-the-fly), which is simple but introduces runtime scaling overhead. On the other hand, Post-Training Static Quantization (PTSQ) quantizes both weights and activations offline using calibration data, avoiding runtime scale computation and achieving the lowest latency.
2. **Model Footprint Reduction**: I learned that mapping 32-bit floating-point values ($float32$) to 8-bit integers ($int8$) yields an approximate **4x** (or ~75%) reduction in model file size, allowing models to fit into cache memory more easily.
3. **Module Fusion**: I learned that fusing contiguous layer pairs like `Conv2d + ReLU` reduces intermediate memory reads/writes, improving cache locality.

---

### Quantization Mathematics
Quantization maps a continuous, high-precision range of values to a discrete, lower-precision integer range. In PyTorch, this mapping is defined by:

$$q = \text{round}\left(\frac{x}{\text{scale}}\right) + \text{zero\_point}$$

To retrieve the approximated floating-point representation (dequantization):

$$\tilde{x} = (q - \text{zero\_point}) \times \text{scale}$$

Where:
- $x$ is the input floating-point value.
- $q$ is the quantized 8-bit integer.
- $\text{scale}$ is a positive floating-point factor scaling the range.
- $\text{zero\_point}$ is an integer shift mapping to floating-point $0.0$.

---

### Module Fusing in Practice
Before applying static quantization, contiguous operations in the feature extractor (specifically `Conv2d` and `ReLU` pairs) are fused into a single unified `ConvReLU2d` operator:

```python
torch.ao.quantization.fuse_modules(
    model.model.feature_extractor,
    [['0', '1'], ['3', '4'], ['6', '7'], ['9', '10']],
    inplace=True
)
```

Fusing eliminates the memory overhead of writing out intermediate activation maps and reading them back for the activation function, speeding up CPU execution significantly.
