import sys

print("=" * 45)
print("  Environment Verification Check")
print("=" * 45)

packages = {
    "torch": "torch",
    "torchvision": "torchvision",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "tqdm": "tqdm",
}

all_ok = True
for module, name in packages.items():
    try:
        m = __import__(module)
        version = getattr(m, "__version__", "installed")
        print(f"  OK  {name:<20} {version}")
    except ImportError:
        print(f"  MISSING  {name}")
        all_ok = False

print()

# CUDA check
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"  CUDA Available : {cuda}")
    if cuda:
        print(f"  CUDA Device    : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version   : {torch.version.cuda}")
    else:
        print("  CUDA Device    : None (CPU only)")
except Exception as e:
    print(f"  CUDA check failed: {e}")

print()
print("=" * 45)
if all_ok:
    print("  All packages installed successfully!")
else:
    print("  Some packages are MISSING. Install them.")
print("=" * 45)
