# Machine Learning Scripts Repository

This repository contains scripts designed to run machine learning tasks, focusing especially on fine-tuning transformer-based models for sequence-to-sequence tasks.

## Installation

**Prerequisites:**

- Ensure you have the required NVIDIA or AMD drivers to support CUDA or ROCm operations, respectively, if you're planning to train on a GPU.
- Your Python version should be >= 3.6.

**Required Packages:**

Install the necessary packages using pip:

```bash
pip install datasets evaluate sacrebleu sentencepiece accelerate sacremoses transformers
```

## Overview of Fine-Tuning

- **Model Initialization**: Begin with a pre-trained model, which has learned valuable features from extensive datasets.
  
- **Data Preparation**: Load your dataset and preprocess it to make it compatible with the model. This may involve tokenization and defining source and target sequences for translation tasks.
  
- **Training Configuration**: Set hyperparameters, such as learning rate, batch size, epochs, etc.
  
- **Training Loop**: The model refines its features based on the feedback it receives from the loss on the training data.
  
- **Evaluation**: After or during training, the model's performance is assessed on unseen datasets.

## VRAM Usage & Optimization

- **Batch Size**: Increasing the batch size will consume more VRAM. If you experience memory constraints, consider reducing the `per_device_train_batch_size` and `per_device_eval_batch_size`.
  
- **Mixed Precision Training**: Enable mixed precision training with `fp16=True`. This approach uses half-precision (16-bit) computations, decreasing memory requirements and potentially speeding up the training process.

- **Gradient Accumulation**: If larger batch sizes lead to memory errors, consider using gradient accumulation by adjusting `gradient_accumulation_steps` in the training arguments.

## GPU Support and Drivers

- **NVIDIA GPUs**: Ensure you have [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) and [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed.
  
- **AMD GPUs**: For AMD GPUs, ensure you have [ROCm drivers](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) installed. Please note that using PyTorch with ROCm requires a specific setup. Check [PyTorch ROCm documentation](https://pytorch.org/docs/master/notes/rocm.html) for more details.
  
- **Mixed Precision Training**: Modern NVIDIA GPUs with Tensor Cores (e.g., NVIDIA Volta, Turing, Ampere architectures) or AMD GPUs supporting BFLOAT16 will benefit significantly from `fp16=True`.

- **Accelerate Library**: This repository employs the `accelerate` library, which simplifies transitioning between CPU and GPU, along with enabling multi-GPU training.

## Further Reading and Resources

- [HuggingFace Documentation](https://huggingface.co/transformers/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
  
---

This README should give a good starting point for your users. Adjustments can be made based on specific requirements and changes in your repository.
