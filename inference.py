"""
MONAI MedNIST Inference Script
Use this script to make predictions on individual images using a trained model.
"""

import argparse
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    ToTensord,
)
from monai.networks.nets import DenseNet121
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Class names matching the training script
CLASS_NAMES = [
    "AbdomenCT",
    "BreastMRI",
    "CXR",
    "ChestCT",
    "Hand",
    "HeadCT",
]

def get_device():
    """Get the best available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("⚠ Using CPU (CUDA not available)")
        return torch.device("cpu")


CONFIG = {
    "num_classes": 6,
    "image_size": (64, 64),
    "device": None,  # Will be set in main()
}


def load_model(model_path, device):
    """Load a trained model."""
    model = DenseNet121(
        spatial_dims=2,
        in_channels=1,
        out_channels=CONFIG["num_classes"],
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_inference_transforms():
    """Get transforms for inference (same as validation)."""
    return Compose([
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=CONFIG["image_size"]),
        ToTensord(keys="image"),
    ])


def predict_image(model, image_path, transforms, device):
    """Make a prediction on a single image."""
    # Load and transform image
    data = {"image": image_path}
    transformed = transforms(data)
    image_tensor = transformed["image"].unsqueeze(0).to(device)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()


def visualize_prediction(image_path, predicted_class, confidence, all_probs):
    """Visualize the image and prediction results."""
    # Load original image for display
    img = Image.open(image_path).convert('L')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'Input Image\nPredicted: {CLASS_NAMES[predicted_class]}\nConfidence: {confidence*100:.2f}%')
    ax1.axis('off')
    
    # Display probability distribution
    y_pos = np.arange(len(CLASS_NAMES))
    ax2.barh(y_pos, all_probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(CLASS_NAMES)
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Probabilities')
    ax2.set_xlim([0, 1])
    
    # Highlight predicted class
    ax2.barh(predicted_class, all_probs[predicted_class], color='green', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    print(f"\nPrediction visualization saved to: prediction_result.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Inference on MedNIST images')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file to classify')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Skip visualization (just print results)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MONAI MedNIST Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    
    # Set device (GPU if available)
    CONFIG["device"] = get_device()
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model, CONFIG["device"])
    print("Model loaded successfully!")
    
    # Get transforms
    transforms = get_inference_transforms()
    
    # Make prediction
    print(f"\nMaking prediction on {args.image}...")
    predicted_class, confidence, all_probs = predict_image(
        model, args.image, transforms, CONFIG["device"]
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Predicted Class: {CLASS_NAMES[predicted_class]}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nAll Class Probabilities:")
    for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, all_probs)):
        marker = " <-- PREDICTED" if i == predicted_class else ""
        print(f"  {class_name}: {prob*100:.2f}%{marker}")
    print("=" * 60)
    
    # Visualize if requested
    if not args.no_visualize:
        visualize_prediction(args.image, predicted_class, confidence, all_probs)


if __name__ == "__main__":
    main()

