# PruneSLU: Efficient On-Device Spoken Language Understanding

Welcome to the **PruneSLU** repository, which houses the models and code associated with the paper *PruneSLU: On-device Spoken Language Understanding with Vocabulary and Structural Pruning*. This project aims to optimize Spoken Language Understanding (SLU) for on-device applications by leveraging advanced pruning techniques to reduce model size without significantly compromising performance.

## Overview

PruneSLU models are designed to deliver efficient SLU capabilities on resource-constrained devices. The models presented here have been selectively pruned and, in some instances, further fine-tuned to achieve an optimal balance between model size and performance. Whether you're working in a low-resource environment or require robust SLU capabilities on a mobile device, these models provide a range of options tailored to different needs.

## Available Models

### 1. [PruneSLU-15M](https://huggingface.co/kodiak619/PruneSLU-15M)
**PruneSLU-15M** is a streamlined version of the [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en) model, pruned and fine-tuned specifically for SLU tasks. With only 15 million parameters, it is highly efficient, making it ideal for deployment in low-resource environments while maintaining strong SLU performance.

### 2. [PruneSLU-30M](https://huggingface.co/kodiak619/PruneSLU-30M)
**PruneSLU-30M** builds on the 15M model, featuring 30 million parameters to provide enhanced SLU capabilities. This model strikes a balance between efficiency and performance, handling more complex SLU tasks while remaining suitable for on-device usage.

### 3. [PruneSLU-15M-Pruned](https://huggingface.co/kodiak619/PruneSLU-15M-Pruned)
**PruneSLU-15M-Pruned** represents the pruned version of the [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en) model without additional fine-tuning. It serves as a baseline for comparison against models that have undergone further retraining, providing insights into the effectiveness of different retraining strategies post-pruning.

### 4. [PruneSLU-30M-Pruned](https://huggingface.co/kodiak619/PruneSLU-30M-Pruned)
**PruneSLU-30M-Pruned** is the pruned version of the 30M model, also without additional fine-tuning. This model allows for the examination of retraining impacts, offering a benchmark for evaluating the benefits of post-pruning optimization.

## Retraining PruneSLU Models

The following steps guide you through retraining the pruned PruneSLU models using the Hugging Face Transformers library.

### Environment Setup

First, set up your environment by installing the required dependencies:

```bash
conda create --name PruneSLU python=3.8
conda activate PruneSLU

pip install -r requirements.txt
```

### Download Pruned Models

Download the pruned models [PruneSLU-15M-Pruned](https://huggingface.co/kodiak619/PruneSLU-15M-Pruned) and [PruneSLU-30M-Pruned](https://huggingface.co/kodiak619/PruneSLU-30M-Pruned) from the links above. Place the downloaded models in the ./pruned_models directory.


### Retraining Process

To begin retraining, execute the following script:

```bash
bash scripts/train.sh
```

This script will initiate the training process using the pruned models, allowing you to further fine-tune them according to your specific SLU tasks and requirements.