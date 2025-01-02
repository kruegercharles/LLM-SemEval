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

# Save in PyTorch format
torch.save(state_dict, "pytorch_model.bin")
