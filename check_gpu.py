import torch
import ultralytics

print(f"Ultralytics Version: {ultralytics.__version__}")
print(f"PyTorch Version: {torch.__version__}")
print("-----------------------------------------")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ SUCCESS! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ WARNING: Using CPU. Training will be slow.")