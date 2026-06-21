# Distributed Training Project

## Overview

This project demonstrates the implementation of a PyTorch image classification workflow with support for distributed training using Torchrun.

The project was developed as part of the ICTCLD401 assessment requirements and includes:

* Model training
* Evaluation metrics
* Experiment logging
* Model checkpointing
* Distributed execution
* Cloud deployment and environment setup

The model is trained on the CIFAR-10 dataset using a ResNet18 architecture and supports both single-device and distributed execution modes.

---

## Project Structure

```text
.
├── train.py
├── run_distributed.sh
├── requirements.txt
├── README.md
├── evidence/
│   ├── part_a/
│   ├── part_b/
│   ├── part_c/
│   └── logs/
├── runs/
└── checkpoints/
```

---

## Environment Requirements

### Operating Systems

* Windows 11 (local development)
* Ubuntu Linux (Azure VM)

### Python

* Python 3.10 or later

### Required Packages

* torch
* torchvision
* numpy
* pandas
* matplotlib
* scikit-learn
* tqdm

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Local Environment Setup

### Create Virtual Environment

```bash
python -m venv ml_env
```

### Activate Virtual Environment

#### Windows

```bash
ml_env\Scripts\activate
```

#### Linux

```bash
source ml_env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Azure VM Environment Setup

### Provision Azure Virtual Machine

Configuration used:

* Azure Virtual Machine
* Ubuntu Linux
* SSH Authentication
* VSCode Remote SSH

### Connect via SSH

```bash
ssh azureuser@<PUBLIC_IP>
```

### Create Virtual Environment

```bash
python3 -m venv ml_env
source ml_env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running Single Device Training

Example:

```bash
python train.py --epochs 5 --batch-size 128 --lr 0.001
```

Expected output:

```text
Using device: cuda
Epoch 1/5
Train Loss: ...
Validation Accuracy: ...
```

---

## Running Distributed Training

Example:

```bash
bash run_distributed.sh --epochs 5 --batch-size 128 --lr 0.001
```

Expected output:

```text
====================================
Distributed Training Launcher
====================================
Detected GPUs: X
Launching torchrun
```

If only one GPU or CPU is available, the script automatically falls back to single-process execution.

---

## Logging

Training metrics are recorded in the runs directory.

Generated outputs include:

* metrics.csv
* log files
* model checkpoints

Example metrics output:

```csv
epoch,train_loss,train_acc,val_loss,val_acc
1,0.8656,0.8619,0.7535,0.9112
2,0.7018,0.9359,0.7189,0.9230
```

---

## Troubleshooting

### Torchrun Not Found

```bash
python -m torch.distributed.run --nproc_per_node=1 train.py
```

### CUDA Not Available

Verify installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

The training script automatically falls back to CPU execution if CUDA is unavailable.

### Out of Memory Errors

Reduce batch size:

```bash
python train.py --batch-size 32
```

### Missing Dependencies

Reinstall requirements:

```bash
pip install -r requirements.txt
```

---

Evidence Folder Organisation
evidence/
├── part_a/
│   ├── classification report.png
│   ├── confusion matrix.png
│   ├── cuda proof.png
│   ├── dataset+batch_verification.png
│   ├── experiment tracking.png
│   └── inference prediction.png
│
├── part_b/
│   ├── confusion matrix distributed launcher.png
│   ├── distributed training launcher 5 epoch logging.png
│   ├── distributed training launcher batch size 64.png
│   ├── distributed training launcher real run.png
│   ├── distributed training launcher.png
│   └── run_distributed.sh local test.png
│
├── part_c/
│   ├── VSCodeConnectAzureSSH.png
│   └── venvcreation.png
│
└── logs/
Assessment Evidence Mapping
Part A – Single Device Training

Evidence Files:

dataset+batch_verification.png
Demonstrates dataset loading and batch configuration.
cuda proof.png
Demonstrates successful CUDA detection and GPU utilisation.
experiment tracking.png
Demonstrates training metric tracking and experiment logging.
confusion matrix.png
Demonstrates model evaluation output.
classification report.png
Demonstrates precision, recall and F1-score reporting.
inference prediction.png
Demonstrates model reload and inference prediction functionality.

Additional Artefacts:

train.py
requirements.txt
metrics.csv
model checkpoint files
Part B – Distributed Training

Evidence Files:

distributed training launcher.png
Demonstrates launcher startup and hardware detection.
distributed training launcher real run.png
Demonstrates successful execution of distributed training.
distributed training launcher 5 epoch logging.png
Demonstrates logging during a distributed training session.
distributed training launcher batch size 64.png
Demonstrates pass-through argument functionality.
run_distributed.sh local test.png
Demonstrates local execution and validation.
confusion matrix distributed launcher.png
Demonstrates evaluation output from distributed execution.

Additional Artefacts:

run_distributed.sh
distributed training logs
metrics.csv
Part C – Documentation and Environment Setup

Evidence Files:

VSCodeConnectAzureSSH.png
Demonstrates successful VSCode Remote SSH connection to Azure VM.
venvcreation.png
Demonstrates creation and activation of a Python virtual environment.

Additional Artefacts:

README.md
requirements.txt
Azure VM configuration
SSH configuration
---

## Unit Mapping Statement

### ICTCLD401 - 4.1 Configure and Prepare a Development Environment

Evidence:

* Azure VM provisioning
* Python virtual environment creation
* Dependency installation
* VSCode Remote SSH configuration

### ICTCLD401 - 4.2 Implement and Validate Environment Configuration

Evidence:

* Successful package installation
* Environment activation
* Script execution validation
* Distributed launcher configuration

### ICTCLD401 - 4.3 Maintain Documentation and Supporting Evidence

Evidence:

* README.md documentation
* Code comments and docstrings
* Organised evidence folder structure
* Training logs and screenshots

---

## Author

Clayton Lin

Certificate IV in Data Science and Artificial Intelligence
