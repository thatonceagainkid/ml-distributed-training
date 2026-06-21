# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:37:20 2026

@author: clayt
"""
#!/bin/bash

echo "===================================="
echo "Distributed Training Launcher"
echo "===================================="

# -------------------------------------
# GPU Detection
# -------------------------------------

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
else
    GPU_COUNT=0
fi

echo "Detected GPUs: $GPU_COUNT"

# -------------------------------------
# Defaults
# -------------------------------------

EPOCHS=10
BATCH_SIZE=32
LR=0.001
TRACKER=none

# -------------------------------------
# Argument Handling
# -------------------------------------

while [[ $# -gt 0 ]]
do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --tracker)
            TRACKER="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# -------------------------------------
# Validation
# -------------------------------------

if [ "$EPOCHS" -le 0 ]; then
    echo "Error: epochs must be > 0"
    exit 1
fi

if [ "$BATCH_SIZE" -le 0 ]; then
    echo "Error: batch-size must be > 0"
    exit 1
fi

# -------------------------------------
# Logging
# -------------------------------------

RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="runs/run_$RUN_ID"

mkdir -p "$LOG_DIR"

echo "Run Directory: $LOG_DIR"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Tracker: $TRACKER"

# -------------------------------------
# Launch
# -------------------------------------

if [ "$GPU_COUNT" -gt 1 ]; then

    echo "Launching torchrun (multi-GPU)"

    torchrun \
        --nproc_per_node=$GPU_COUNT \
        train.py \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --output-dir $LOG_DIR

else

    echo "Single GPU / CPU fallback"

    python train.py \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --output-dir $LOG_DIR

fi

echo "Run Complete"