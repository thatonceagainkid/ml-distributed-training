# ML Distributed Training Project

## Overview

This project demonstrates the development, training, evaluation, and deployment of a machine learning image classification model using PyTorch. The project includes single-node model training, distributed training using `torchrun`, experiment tracking, model evaluation, and cloud-based execution on an Azure Virtual Machine.

The project was completed as part of the requirements for:

* ICTCLD401 – Develop and Deploy Machine Learning Solutions
* ICTCLD402 – Manage Machine Learning Data
* ICTCLD403 – Implement Machine Learning Models

---

# Project Objectives

The objectives of this project were to:

* Build an image classification model using the CIFAR-10 dataset.
* Implement a reproducible machine learning workflow.
* Train and evaluate the model using PyTorch.
* Implement distributed training using PyTorch Distributed Data Parallel (DDP).
* Deploy and execute the project within a cloud environment.
* Document setup, execution, and troubleshooting procedures.

---

# Project Structure

```text
ml-distributed-training/
│
├── data/
├── evidence/
│   ├── PartA/
│   ├── PartB/
│   └── PartC/
│
├── notebooks/
├── outputs/
│   └── metrics.csv
│
├── train.py
├── run_distributed.sh
├── print_env.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Environment Requirements

## Operating System

* Ubuntu 24.04 LTS (Azure VM)

## Python

* Python 3.12+

## Required Packages

Installed through:

```bash
pip install -r requirements.txt
```

Main dependencies include:

* torch
* torchvision
* pandas
* numpy
* matplotlib
* scikit-learn
* tqdm

---

# Azure Environment Setup

## Step 1 – Provision Azure VM

Create an Azure Virtual Machine using Ubuntu 24.04 LTS.

Recommended specifications:

* 2+ vCPUs
* 8GB RAM
* Internet connectivity enabled
* SSH access enabled

Evidence:

* VM Creation.png

---

## Step 2 – Connect Using VSCode Remote SSH

Install the VSCode Remote SSH extension.

Connect to the Azure VM:

```bash
ssh -i ml-distributed-training_key.pem azureuser@<public-ip>
```

Evidence:

* VSCodeConnectAzureSSH.png

---

## Step 3 – Create Python Virtual Environment

Create and activate a virtual environment:

```bash
python3 -m venv ml_env
source ml_env/bin/activate
```

Evidence:

* venvcreation.png

---

## Step 4 – Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Evidence:

* dependency install pt1.png
* dependency install pt2.png

---

# Dataset

The project uses the CIFAR-10 image classification dataset.

Dataset classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

Dataset loading is handled automatically by PyTorch:

```python
torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)
```

Downloaded datasets are cached locally and reused in subsequent executions.

---

# Running the Training Script

Execute model training:

```bash
python train.py
```

Example:

```bash
python train.py --epochs 5 --batch-size 64
```

Expected output:

```text
Using device: cuda
Epoch 1/5
Training Loss: ...
Validation Accuracy: ...
```

Evidence:

* VM shell run.png

---

# Running Distributed Training

Execute distributed training:

```bash
bash run_distributed.sh
```

Example output:

```text
====================================
Distributed Training Launcher
====================================
Detected GPUs: 2
Launching torchrun...
```

Evidence:

* VM distribution.sh run.png

---

# Output Files

Training generates:

```text
outputs/
└── metrics.csv
```

The metrics file records:

* Epoch
* Training Loss
* Validation Loss
* Accuracy

---

# Troubleshooting

## Virtual Environment Not Activated

Check:

```bash
which python
```

Expected:

```text
.../ml_env/bin/python
```

---

## Missing Packages

Reinstall dependencies:

```bash
pip install -r requirements.txt
```

---

## CUDA Not Detected

Verify installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected:

```text
True
```

or

```text
False
```

depending on VM hardware configuration.

---

## Distributed Script Permission Error

Grant execution permission:

```bash
chmod +x run_distributed.sh
```

---

# Evidence Folder

## Part A – Model Development and Evaluation

* classification report.png
* confusion matrix.png
* cuda proof.png
* dataset+batch_verification.png
* experiment tracking.png
* inference prediction.png

---

## Part B – Distributed Training

* confusion matrix distributed launcher.png
* distributed training launcher.png
* distributed training launcher real run.png
* distributed training launcher batch size 64.png
* distributed training launcher 5 epoch logging.png
* run_distributed.sh local test.png

---

## Part C – Environment Setup and Documentation

* VM Creation.png
* VSCodeConnectAzureSSH.png
* venvcreation.png
* dependency install pt1.png
* dependency install pt2.png
* VM shell run.png
* VM distribution.sh run.png

---

# Mapping to Unit Criteria

## ICTCLD401 – Environment Configuration

### 4.1 Configure Development Environment

Evidence:

* VM Creation.png
* VSCodeConnectAzureSSH.png
* venvcreation.png

Activities:

* Provisioned Azure VM
* Configured SSH connectivity
* Created Python virtual environment

### 4.2 Install and Configure Software Components

Evidence:

* dependency install pt1.png
* dependency install pt2.png
* VM shell run.png

Activities:

* Installed project dependencies
* Configured PyTorch environment
* Verified successful execution

### 4.3 Document and Maintain Environment

Evidence:

* README.md
* requirements.txt
* GitHub repository
* Evidence folder

Activities:

* Produced setup documentation
* Recorded execution procedures
* Organised assessment evidence
* Maintained reproducible project structure

---

# Author

Clayton Lin

Certificate IV in Data Science and Artificial Intelligence

2026
