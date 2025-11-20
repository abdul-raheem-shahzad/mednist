"""
GPU Check Script
Use this script to verify if your system has GPU support and PyTorch can use it.
"""

import sys

print("=" * 60)
print("GPU and PyTorch Configuration Check")
print("=" * 60)

# Check PyTorch installation
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
except ImportError:
    print("✗ PyTorch is not installed!")
    sys.exit(1)

# Check CUDA availability
print(f"\nCUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print("\n" + "=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
    
    # Test GPU computation
    print("\n" + "=" * 60)
    print("GPU TEST")
    print("=" * 60)
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU computation test: SUCCESS")
        print(f"  Result tensor device: {z.device}")
        print(f"  Result shape: {z.shape}")
    except Exception as e:
        print(f"✗ GPU computation test: FAILED")
        print(f"  Error: {e}")
else:
    print("\n" + "=" * 60)
    print("⚠ CUDA NOT AVAILABLE")
    print("=" * 60)
    print("PyTorch is installed but cannot detect CUDA/GPU.")
    print("\nPossible reasons:")
    print("1. You installed CPU-only PyTorch")
    print("2. NVIDIA GPU drivers are not installed")
    print("3. CUDA toolkit is not installed")
    print("\nTo fix:")
    print("1. Check if you have NVIDIA GPU: Run 'nvidia-smi' in terminal")
    print("2. If GPU exists, install PyTorch with CUDA:")
    print("   Visit: https://pytorch.org/get-started/locally/")
    print("   Example: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 60)

