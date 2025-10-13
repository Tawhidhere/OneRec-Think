#!/bin/bash

# Setup conda environment for OneRec-Think training
# This script creates a conda environment with all dependencies needed to run the training pipeline

set -e

ENV_NAME="onerec-think"
PYTHON_VERSION="3.10"

echo "========================================"
echo "Setting up conda environment: ${ENV_NAME}"
echo "========================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment: ${ENV_NAME}"
    conda env remove -n ${ENV_NAME} -y
fi

# Create new conda environment
echo "Creating new conda environment with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust CUDA version as needed)
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and related packages
echo "Installing transformers and HuggingFace packages..."
pip install transformers>=4.36.0
pip install datasets
pip install accelerate
pip install huggingface-hub

# Install PEFT for LoRA training
echo "Installing PEFT..."
pip install peft

# Install DeepSpeed for distributed training
echo "Installing DeepSpeed..."
pip install deepspeed

# Install vLLM for fast inference
echo "Installing vLLM..."
pip install vllm

# Install data processing libraries
echo "Installing data processing libraries..."
pip install pandas
pip install pyarrow
pip install numpy

# Install utility packages
echo "Installing utility packages..."
pip install tqdm
pip install sentencepiece
pip install protobuf

# Verify installations
echo ""
echo "========================================"
echo "Verifying installations..."
echo "========================================"

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import peft; print(f'PEFT version: {peft.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"

echo ""
echo "========================================"
echo "Environment setup complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "    conda activate ${ENV_NAME}"
