# MONAI MedNIST Classification Project

This project demonstrates how to use MONAI (Medical Open Network for AI) to train a deep learning model for medical image classification using the MedNIST dataset.

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
2. Display GPU information if CUDA is available
3. Automatically download the MedNIST dataset (if not already present)
4. Split the data into training (80%) and validation (20%) sets
5. Create data loaders with appropriate transforms
6. Train a DenseNet121 model for 10 epochs
7. Save the best model based on validation accuracy
8. Generate a training history plot

### Configuration

You can modify the training parameters in `train.py` by editing the `CONFIG` dictionary:

```python
CONFIG = {
    "data_dir": "./MedNIST",
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "num_classes": 6,
    "image_size": (64, 64),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
```

### Inference

After training, you can use the inference script to test the model on new images:

```bash
python inference.py --model best_model.pth --image path/to/image.jpeg
```

## Key MONAI Concepts Demonstrated

### 1. **Data Loading**
- Using MONAI's `Dataset` class with data dictionaries
- Custom data transforms pipeline

### 2. **Transforms**
- **LoadImage**: Loads medical images
- **EnsureChannelFirst**: Ensures channel dimension is first (for grayscale images)
- **ScaleIntensity**: Normalizes pixel values
- **Resize**: Resizes images to a fixed size
- **RandRotate, RandFlip, RandZoom**: Data augmentation for training

### 3. **Model Architecture**
- **DenseNet121**: A pre-built MONAI network for 2D classification
- Configured for grayscale input (1 channel) and 6 output classes

### 4. **Training Pipeline**
- Standard PyTorch training loop with MONAI components
- Progress bars with real-time metrics
- Model checkpointing

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
- Training accuracy: ~95-98% after 10 epochs
- Validation accuracy: ~95-97%
- The model learns to distinguish between different medical imaging modalities

## GPU Usage

The code **automatically uses GPU if available**. When you run `train.py`, it will:
- Detect if CUDA/GPU is available
- Display GPU information (name, memory, CUDA version)
- Use GPU for all model operations (training and inference)
- Fall back to CPU if GPU is not available

**To ensure GPU is used:**
1. Install PyTorch with CUDA support (see Installation section)
2. Run `python check_gpu.py` to verify
3. When training, you should see GPU information printed at the start

**Performance**: Training on GPU is typically **10-50x faster** than CPU, depending on your hardware.

## Tips for Learning

1. **Experiment with transforms**: Try different augmentation strategies
2. **Modify the model**: Try different architectures (ResNet, EfficientNet, etc.)
3. **Adjust hyperparameters**: Change learning rate, batch size, epochs
4. **Visualize**: Use the inference script to see predictions on individual images
5. **Compare GPU vs CPU**: Try training with and without GPU to see the speed difference

## Resources

- [MONAI Documentation](https://docs.monai.io/)
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
- [PyTorch Documentation](https://pytorch.org/docs/)

## License

This project is for educational purposes. The MedNIST dataset is provided by MONAI for research and educational use.

