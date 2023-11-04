import torch
print (f"devices found: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"{i}: {torch.cuda.get_device_name(i)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a tensor on the CPU
x = torch.randn(3, 3)

# Move the tensor to the GPU
device = torch.device("cuda:0")

for i in range(100):
    x = x.to(device) # Set to use first found CUDA GPU
    print (x)

import onnxruntime as ort

print(ort.get_device())