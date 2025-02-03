import torch
import os

"""
When downloading Llama models from the Llama website, the model is saved in a .pth file, but we need a .bin file.
"""

PATH = "models/checkpoints/Llama3.1-8B-Instruct/consolidated.00.pth"

# check if the file exists
if not os.path.exists(PATH):
    raise FileNotFoundError(f"Model file {PATH} not found")

# Load the .pth file
state_dict = torch.load(PATH)

dtypes = {tensor.dtype for tensor in state_dict.values()}
print(f"\nDtypes used: {dtypes}")

# Get shapes of all tensors
shapes = {key: tensor.shape for key, tensor in state_dict.items()}
print("\nTensor shapes:")
for key, shape in shapes.items():
    print(f"{key}: {shape}")

# Get memory usage
total_params = sum(tensor.numel() for tensor in state_dict.values())
memory_bytes = sum(
    tensor.element_size() * tensor.numel() for tensor in state_dict.values()
)
print(f"\nTotal parameters: {total_params:,}")
print(f"Memory usage: {memory_bytes / 1024**2:.2f} MB")
