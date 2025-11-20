"""
Data Exploration Script
Use this script to explore the MedNIST dataset structure and visualize sample images.
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

CLASS_NAMES = [
    "AbdomenCT",
    "BreastMRI",
    "CXR",
    "ChestCT",
    "Hand",
    "HeadCT",
]

DATA_DIR = "./MedNIST"


def explore_dataset(data_dir):
    """Explore the dataset structure and show statistics."""
    print("=" * 60)
    print("MedNIST Dataset Exploration")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"Dataset not found at {data_dir}")
        print("Run train.py first to download the dataset.")
        return
    
    total_images = 0
    class_counts = {}
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) 
                     if f.endswith(('.jpeg', '.jpg', '.png'))]
            count = len(images)
            class_counts[class_name] = count
            total_images += count
            print(f"{class_name:15s}: {count:5d} images")
        else:
            class_counts[class_name] = 0
            print(f"{class_name:15s}: Not found")
    
    print("-" * 60)
    print(f"{'Total':15s}: {total_images:5d} images")
    print("=" * 60)
    
    return class_counts


def visualize_samples(data_dir, samples_per_class=3):
    """Visualize sample images from each class."""
    print("\n" + "=" * 60)
    print("Visualizing Sample Images")
    print("=" * 60)
    
    fig, axes = plt.subplots(len(CLASS_NAMES), samples_per_class, 
                            figsize=(samples_per_class * 3, len(CLASS_NAMES) * 3))
    
    if len(CLASS_NAMES) == 1:
        axes = axes.reshape(1, -1)
    
    for row, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            for col in range(samples_per_class):
                axes[row, col].axis('off')
            continue
        
        # Get sample images
        images = [f for f in os.listdir(class_dir) 
                 if f.endswith(('.jpeg', '.jpg', '.png'))]
        images = images[:samples_per_class]
        
        for col, img_name in enumerate(images):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert('L')
            
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
            
            if col == 0:
                axes[row, col].set_ylabel(class_name, rotation=0, 
                                         labelpad=20, fontsize=10)
    
    plt.suptitle('MedNIST Dataset - Sample Images from Each Class', 
                fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    print("Sample visualization saved to: dataset_samples.png")
    plt.show()


def show_image_statistics(data_dir):
    """Show statistics about image sizes and pixel values."""
    print("\n" + "=" * 60)
    print("Image Statistics")
    print("=" * 60)
    
    sizes = []
    pixel_ranges = []
    
    # Sample images from each class
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        images = [f for f in os.listdir(class_dir) 
                 if f.endswith(('.jpeg', '.jpg', '.png'))]
        
        # Sample up to 10 images per class
        for img_name in images[:10]:
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert('L')
            img_array = np.array(img)
            
            sizes.append(img.size)  # (width, height)
            pixel_ranges.append((img_array.min(), img_array.max()))
    
    if sizes:
        sizes = np.array(sizes)
        print(f"Image sizes (width x height):")
        print(f"  Min: {sizes[:, 0].min()} x {sizes[:, 1].min()}")
        print(f"  Max: {sizes[:, 0].max()} x {sizes[:, 1].max()}")
        print(f"  Mean: {sizes[:, 0].mean():.1f} x {sizes[:, 1].mean():.1f}")
        
        pixel_ranges = np.array(pixel_ranges)
        print(f"\nPixel value ranges:")
        print(f"  Min pixel value: {pixel_ranges[:, 0].min()}")
        print(f"  Max pixel value: {pixel_ranges[:, 1].max()}")
        print(f"  Mean min: {pixel_ranges[:, 0].mean():.1f}")
        print(f"  Mean max: {pixel_ranges[:, 1].mean():.1f}")


def main():
    # Explore dataset structure
    class_counts = explore_dataset(DATA_DIR)
    
    if class_counts and sum(class_counts.values()) > 0:
        # Visualize samples
        visualize_samples(DATA_DIR, samples_per_class=3)
        
        # Show statistics
        show_image_statistics(DATA_DIR)
    else:
        print("\nPlease run train.py first to download the dataset.")


if __name__ == "__main__":
    main()

