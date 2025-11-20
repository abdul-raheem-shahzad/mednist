# MONAI MedNIST Classification Project

This project demonstrates how to use MONAI (Medical Open Network for AI) to train a deep learning model for medical image classification using the MedNIST dataset. The implementation uses **DenseNet121** architecture with advanced GPU optimizations including mixed precision training, gradient accumulation, early stopping, and learning rate scheduling.

## What's Included

This project implements a complete medical image classification pipeline with:

- **Model**: DenseNet121 (MONAI implementation)
- **Dataset**: MedNIST (6 classes of medical images)
- **Training Features**:
  - Mixed precision training (FP16) for GPU efficiency
  - Gradient accumulation (effective batch size: 8)
  - Early stopping (patience: 15 epochs)
  - Learning rate scheduling (ReduceLROnPlateau)
  - Model checkpointing (saves best model)
  - Training history visualization
- **GPU Optimizations**: CUDA 12.6 support, memory management, error recovery
- **Platform Support**: Windows-optimized data loading

## What is MedNIST?

MedNIST is a medical imaging dataset containing 6 classes of medical images:
- **AbdomenCT**: Abdominal CT scans
- **BreastMRI**: Breast MRI images
- **CXR**: Chest X-rays
- **ChestCT**: Chest CT scans
- **Hand**: Hand X-rays
- **HeadCT**: Head CT scans

The dataset is automatically downloaded when you run the training script.

## Project Structure

```
mednist/
├── train.py              # Main training script
├── inference.py          # Inference script for testing trained models
├── explore_data.py       # Data exploration and visualization
├── check_gpu.py          # GPU verification script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

### 1. Install PyTorch with GPU Support (Recommended)

**Important**: The default `pip install torch` installs CPU-only version. For NVIDIA GPU support, you need to install PyTorch with CUDA.

**Check your CUDA version first:**
```bash
nvidia-smi
```

**Then install PyTorch with matching CUDA version:**

For CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 12.6:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Or visit [PyTorch Installation](https://pytorch.org/get-started/locally/) to get the exact command for your system.

**Verify GPU support:**
```bash
python check_gpu.py
```

### 2. Install Other Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you already installed CPU-only PyTorch, uninstall it first:
```bash
pip uninstall torch torchvision
```
Then install the CUDA version as shown above.

## Usage

### Verify GPU Setup

Before training, check if GPU is available:

```bash
python check_gpu.py
```

This will show:
- PyTorch version
- CUDA availability
- GPU information (name, memory, etc.)
- GPU computation test

### Training

Run the training script:

```bash
python train.py
```

The script will:
1. **Automatically detect and use GPU** if available (falls back to CPU if not)
2. Display GPU information if CUDA is available (GPU name, CUDA version, memory)
3. Automatically download the MedNIST dataset (if not already present)
4. Split the data into training (80%) and validation (20%) sets
5. Create data loaders with appropriate transforms and augmentation
6. Train a **DenseNet121** model with advanced optimizations:
   - Mixed precision training (FP16) for GPU efficiency
   - Gradient accumulation for effective larger batch sizes
   - Early stopping to prevent overfitting
   - Learning rate scheduling (ReduceLROnPlateau)
   - Model checkpointing (saves best model based on validation accuracy)
7. Generate a training history plot (loss and accuracy curves)
8. Display GPU memory usage statistics

### Configuration

You can modify the training parameters in `train.py` by editing the `CONFIG` dictionary:

```python
CONFIG = {
    "data_dir": "./MedNIST",
    "batch_size": 4,  # Reduced for GPU memory efficiency
    "gradient_accumulation_steps": 2,  # Effective batch size = 4 * 2 = 8
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "num_classes": 6,
    "image_size": (64, 64),
    "device": None,  # Automatically set based on availability
    "use_mixed_precision": True,  # Use FP16 to reduce memory by ~50%
    "early_stopping_patience": 15,  # Stop if no improvement for 15 epochs
    "reduce_lr_patience": 7,  # Reduce LR if no improvement for 7 epochs
}
```

**Key Configuration Features:**
- **Batch Size**: 4 (with gradient accumulation of 2 steps = effective batch size of 8)
- **Epochs**: 100 (with early stopping to prevent overfitting)
- **Mixed Precision**: Enabled by default (FP16) for GPU memory efficiency
- **Early Stopping**: Stops training if validation loss doesn't improve for 15 epochs
- **Learning Rate Scheduling**: Automatically reduces learning rate by 50% if no improvement for 7 epochs

### Inference

After training, you can use the inference script to test the model on new images:

```bash
python inference.py --model best_model.pth --image path/to/image.jpeg
```

## Key Features and Technologies Used

### 1. **Model Architecture**
- **DenseNet121**: Pre-built MONAI network for 2D medical image classification
- Configured for grayscale input (1 channel) and 6 output classes
- Spatial dimensions: 2D
- Input size: 64x64 pixels

### 2. **Data Loading and Processing**
- Using MONAI's `Dataset` class with data dictionaries
- Custom data transforms pipeline with augmentation
- Train/validation split: 80/20
- Windows-optimized data loading (num_workers=0, pin_memory=False)

### 3. **Data Transforms**
- **LoadImaged**: Loads medical images from disk
- **EnsureChannelFirstd**: Ensures channel dimension is first (for grayscale images)
- **ScaleIntensityd**: Normalizes pixel values to [0, 1]
- **Resized**: Resizes images to fixed dimensions (64x64)
- **RandRotated**: Random rotation augmentation (±15 degrees, 50% probability)
- **RandFlipd**: Random horizontal flip (50% probability)
- **RandZoomd**: Random zoom (0.9-1.1x, 50% probability)
- **ToTensord**: Converts to PyTorch tensors

### 4. **Training Optimizations**
- **Mixed Precision Training (FP16)**: Reduces GPU memory usage by ~50% and speeds up training
- **Gradient Accumulation**: Accumulates gradients over multiple batches (effective batch size = 8)
- **Early Stopping**: Prevents overfitting by stopping when validation loss plateaus (patience: 15 epochs)
- **Learning Rate Scheduling**: ReduceLROnPlateau reduces LR by 50% when validation loss plateaus (patience: 7 epochs)
- **Model Checkpointing**: Automatically saves the best model based on validation accuracy
- **GPU Memory Management**: Periodic cache clearing to prevent out-of-memory errors
- **CUDA Error Handling**: Graceful handling of CUDA errors with automatic recovery

### 5. **Training Pipeline**
- Adam optimizer with learning rate 1e-4
- CrossEntropyLoss for multi-class classification
- Progress bars with real-time metrics (loss, accuracy)
- Training history visualization (loss and accuracy plots)
- Reproducibility: Random seed set to 42 for deterministic results

## Understanding the Code

### Data Dictionary Format

MONAI uses a list of dictionaries to represent datasets. Each dictionary contains:
```python
{
    "image": "path/to/image.jpeg",
    "label": 0  # Class index (0-5)
}
```

### Transform Pipeline

Transforms are applied in sequence:
1. Load the image from disk
2. Add channel dimension (grayscale → [1, H, W])
3. Normalize pixel values to [0, 1]
4. Resize to fixed dimensions
5. Apply augmentation (training only)
6. Convert to PyTorch tensor

### Model Output

The model outputs logits for 6 classes. Use `torch.max()` to get the predicted class:
```python
outputs = model(images)  # Shape: [batch_size, 6]
_, predicted = torch.max(outputs, 1)  # Get class indices
```

## Expected Results

With default settings, you should see:
- Training accuracy: ~95-98% after training completes
- Validation accuracy: ~95-97%
- The model learns to distinguish between different medical imaging modalities
- Training typically stops early (before 100 epochs) due to early stopping mechanism
- Best model is automatically saved when validation accuracy improves
- Training history plots saved to `training_history.png`

## GPU Usage and Optimizations

The code **automatically uses GPU if available** with advanced optimizations. When you run `train.py`, it will:
- Detect if CUDA/GPU is available
- Display detailed GPU information:
  - GPU device name
  - CUDA version
  - Number of GPUs
  - Total GPU memory
  - Memory allocation and reservation
- Use GPU for all model operations (training and inference)
- Fall back to CPU if GPU is not available

**GPU Optimizations Implemented:**
1. **Mixed Precision (FP16)**: Reduces memory usage by ~50% and increases training speed
2. **Gradient Accumulation**: Allows effective larger batch sizes without increasing memory
3. **Memory Management**: Periodic CUDA cache clearing to prevent out-of-memory errors
4. **Error Recovery**: Automatic handling and recovery from CUDA errors
5. **Windows Compatibility**: Optimized settings for Windows (num_workers=0, pin_memory=False)

**To ensure GPU is used:**
1. Install PyTorch with CUDA support matching your CUDA version (see Installation section)
2. For CUDA 12.6, use: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
3. Run `python check_gpu.py` to verify GPU availability
4. When training, you should see GPU information printed at the start

**Performance**: Training on GPU is typically **10-50x faster** than CPU, depending on your hardware. With mixed precision enabled, you can expect even better performance and lower memory usage.

## Project Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework (with CUDA 12.6 support)
- **MONAI** (>=1.3.0): Medical imaging AI framework
- **NumPy** (>=1.21.0): Numerical computing
- **Matplotlib** (>=3.5.0): Plotting and visualization
- **Pillow** (>=9.0.0): Image processing
- **tqdm** (>=4.64.0): Progress bars
- **requests** (>=2.28.0): HTTP library for dataset download

### Key MONAI Components Used
- `monai.apps.download_and_extract`: Dataset download
- `monai.data.Dataset, DataLoader`: Data loading
- `monai.transforms`: Image preprocessing and augmentation
- `monai.networks.nets.DenseNet121`: Model architecture
- `monai.utils.set_determinism`: Reproducibility

### PyTorch Components Used
- `torch.optim.Adam`: Optimizer
- `torch.optim.lr_scheduler.ReduceLROnPlateau`: Learning rate scheduler
- `torch.nn.CrossEntropyLoss`: Loss function
- `torch.cuda.amp.GradScaler, autocast`: Mixed precision training

## Tips for Learning

1. **Experiment with transforms**: Try different augmentation strategies and probabilities
2. **Modify the model**: Try different architectures (ResNet, EfficientNet, etc.) from MONAI
3. **Adjust hyperparameters**: Change learning rate, batch size, epochs, early stopping patience
4. **Visualize**: Use the inference script to see predictions on individual images
5. **Compare GPU vs CPU**: Try training with and without GPU to see the speed difference
6. **Monitor GPU memory**: Watch GPU memory usage to optimize batch size and gradient accumulation
7. **Experiment with mixed precision**: Disable it to see the difference in memory usage and speed

## Resources

- [MONAI Documentation](https://docs.monai.io/)
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
- [PyTorch Documentation](https://pytorch.org/docs/)

## License

This project is for educational purposes. The MedNIST dataset is provided by MONAI for research and educational use.

