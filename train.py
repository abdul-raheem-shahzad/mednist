"""
MONAI MedNIST Training Script
This script demonstrates how to use MONAI to train a classification model on the MedNIST dataset.
MedNIST contains 6 classes of medical imaging data.
"""

import os
import torch
import numpy as np
from monai.apps import download_and_extract
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    ToTensord,
    RandRotated,
    RandFlipd,
    RandZoomd,
)
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Set random seed for reproducibility
set_determinism(seed=42)


def get_device():
    """
    Get the best available device (GPU if available, else CPU).
    Also prints GPU information if CUDA is available.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ CUDA is available!")
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return device
    else:
        print("⚠ CUDA is not available. Using CPU.")
        print("  To use GPU, install PyTorch with CUDA support:")
        print("  Visit: https://pytorch.org/get-started/locally/")
        return torch.device("cpu")


# Configuration - GPU Efficient
CONFIG = {
    "data_dir": "./MedNIST",
    "batch_size": 4,  # Reduced for GPU memory efficiency
    "gradient_accumulation_steps": 2,  # Effective batch size = 4 * 2 = 8
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "num_classes": 6,
    "image_size": (64, 64),
    "device": None,  # Will be set in main()
    "use_mixed_precision": True,  # Use FP16 to reduce memory by ~50%
    "early_stopping_patience": 15,  # Stop if no improvement for 15 epochs
    "reduce_lr_patience": 7,  # Reduce LR if no improvement for 7 epochs
}

# MedNIST class names
CLASS_NAMES = [
    "AbdomenCT",
    "BreastMRI",
    "CXR",
    "ChestCT",
    "Hand",
    "HeadCT",
]


def download_mednist(data_dir):
    """Download MedNIST dataset if not already present."""
    if not os.path.exists(data_dir):
        print(f"Downloading MedNIST dataset to {data_dir}...")
        download_and_extract(
            url="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz",
            filepath="./MedNIST.tar.gz",
            output_dir="./",
        )
        print("Download complete!")
    else:
        print(f"MedNIST dataset already exists at {data_dir}")


def get_data_dicts(data_dir, split="train"):
    """
    Create data dictionary list for MONAI Dataset.
    Each dictionary contains the path to an image and its label.
    """
    data_dicts = []
    
    # MedNIST structure: data_dir/class_name/*.jpeg
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for filename in os.listdir(class_dir):
            if filename.endswith(('.jpeg', '.jpg', '.png')):
                data_dicts.append({
                    "image": os.path.join(class_dir, filename),
                    "label": class_idx,
                })
    
    # Simple train/val split (80/20)
    np.random.shuffle(data_dicts)
    split_idx = int(len(data_dicts) * 0.8)
    
    if split == "train":
        return data_dicts[:split_idx]
    else:
        return data_dicts[split_idx:]


def get_transforms(mode="train"):
    """
    Define data transforms for training and validation.
    Training includes augmentation, validation does not.
    """
    if mode == "train":
        return Compose([
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityd(keys="image"),
            Resized(keys="image", spatial_size=CONFIG["image_size"]),
            RandRotated(keys="image", range_x=15, prob=0.5),
            RandFlipd(keys="image", spatial_axis=0, prob=0.5),
            RandZoomd(keys="image", min_zoom=0.9, max_zoom=1.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ])
    else:
        return Compose([
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityd(keys="image"),
            Resized(keys="image", spatial_size=CONFIG["image_size"]),
            ToTensord(keys=["image", "label"]),
        ])


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_mixed_precision=False, gradient_accumulation_steps=1):
    """Train for one epoch with mixed precision and gradient accumulation for GPU efficiency."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()  # Zero gradients at the start
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        try:
            images = batch["image"].to(device, non_blocking=False)
            labels = batch["label"].long().to(device, non_blocking=False)
            
            # Forward pass with mixed precision (FP16) for GPU efficiency
            if use_mixed_precision and scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Update weights only after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Statistics
            loss_value = loss.item() * gradient_accumulation_steps  # Unscale for display
            total_loss += loss_value
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "acc": f"{100 * correct / total:.2f}%"
            })
            
            # Clear cache periodically to free GPU memory
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"\n⚠ CUDA Error detected: {e}")
                print("Clearing CUDA cache and continuing...")
                torch.cuda.empty_cache()
                continue
            else:
                raise
    
    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device, use_mixed_precision=False):
    """Validate the model with mixed precision support for GPU efficiency."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            try:
                images = batch["image"].to(device, non_blocking=False)
                labels = batch["label"].long().to(device, non_blocking=False)
                
                # Use mixed precision for validation too
                if use_mixed_precision:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                loss_value = loss.item()
                total_loss += loss_value
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "acc": f"{100 * correct / total:.2f}%"
                })
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"\n⚠ CUDA Error detected: {e}")
                    print("Clearing CUDA cache and continuing...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
    
    return total_loss / len(dataloader), 100 * correct / total


# Callback Classes
class EarlyStopping:
    """Early stopping callback to stop training if validation loss doesn't improve."""
    def __init__(self, patience=15, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False  # Continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True  # Stop training
            return False  # Continue training


class ModelCheckpoint:
    """Callback to save model checkpoints."""
    def __init__(self, filepath='best_model.pth', monitor='val_acc', mode='max', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        
    def __call__(self, metrics, model, epoch):
        current_value = metrics.get(self.monitor, None)
        if current_value is None:
            return False
            
        is_better = False
        if self.mode == 'max':
            is_better = current_value > self.best_value
        else:
            is_better = current_value < self.best_value
            
        if is_better:
            self.best_value = current_value
            if self.save_best_only:
                torch.save(model.state_dict(), self.filepath)
                return True
        elif not self.save_best_only:
            # Save checkpoint for every epoch
            checkpoint_path = self.filepath.replace('.pth', f'_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            
        return is_better


def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to training_history.png")


def main():
    print("=" * 60)
    print("MONAI MedNIST Classification Training")
    print("=" * 60)
    
    # Set device (GPU if available)
    CONFIG["device"] = get_device()
    print(f"\nUsing device: {CONFIG['device']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Number of epochs: {CONFIG['num_epochs']}")
    print("=" * 60)
    
    # Clear CUDA cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # Download dataset
    download_mednist(CONFIG["data_dir"])
    
    # Get data dictionaries
    print("\nLoading data...")
    train_data = get_data_dicts(CONFIG["data_dir"], split="train")
    val_data = get_data_dicts(CONFIG["data_dir"], split="val")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create datasets and dataloaders
    train_dataset = Dataset(
        data=train_data,
        transform=get_transforms(mode="train")
    )
    val_dataset = Dataset(
        data=val_data,
        transform=get_transforms(mode="val")
    )
    
    # Use num_workers=0 on Windows to avoid CUDA multiprocessing issues
    import platform
    num_workers = 0 if platform.system() == "Windows" else 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory to avoid CUDA issues on Windows
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory to avoid CUDA issues on Windows
    )
    
    # Create model - DenseNet121
    print("\nCreating model...")
    model = DenseNet121(
        spatial_dims=2,
        in_channels=1,
        out_channels=CONFIG["num_classes"],
    ).to(CONFIG["device"])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=CONFIG["reduce_lr_patience"]
    )
    
    # Mixed precision scaler for GPU efficiency
    scaler = GradScaler() if CONFIG["use_mixed_precision"] and torch.cuda.is_available() else None
    
    # Initialize callbacks
    early_stopping = EarlyStopping(
        patience=CONFIG["early_stopping_patience"],
        restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        filepath="best_model.pth",
        monitor='val_acc',
        mode='max',
        save_best_only=True
    )
    
    # Training history
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    # Training loop
    print("\nStarting training...")
    print(f"Using mixed precision (FP16): {CONFIG['use_mixed_precision'] and scaler is not None}")
    print(f"Gradient accumulation steps: {CONFIG['gradient_accumulation_steps']}")
    print(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print("=" * 60)
    
    start_time = time.time()
    best_val_acc = 0.0
    
    for epoch in range(CONFIG["num_epochs"]):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        print("-" * 60)
        
        # Train with GPU optimizations
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, CONFIG["device"],
            scaler=scaler,
            use_mixed_precision=CONFIG["use_mixed_precision"] and scaler is not None,
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"]
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate with mixed precision
        val_loss, val_acc = validate(
            model, val_loader, criterion, CONFIG["device"],
            use_mixed_precision=CONFIG["use_mixed_precision"] and scaler is not None
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Callbacks
        metrics = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'train_acc': train_acc
        }
        
        # Model checkpoint callback
        is_best = model_checkpoint(metrics, model, epoch + 1)
        if is_best:
            best_val_acc = val_acc
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping callback
        should_stop = early_stopping(val_loss, model)
        if should_stop:
            print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
            print(f"  Best validation loss: {early_stopping.best_loss:.4f}")
            break
        
        # Clear cache after each epoch to free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total training time: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    if train_accs:
        print(f"Final Training Accuracy: {train_accs[-1]:.2f}%")
    if val_accs:
        print(f"Final Validation Accuracy: {val_accs[-1]:.2f}%")
    print("=" * 60)
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    print("\nModel saved to: best_model.pth")
    print("Training history saved to: training_history.png")
    
    # Final GPU memory info
    if torch.cuda.is_available():
        print(f"\nFinal GPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()

