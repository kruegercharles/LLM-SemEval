import torch
import os

"""
When downloading Llama models from the Llama website, the model is saved in a .pth file, but we need a .bin file.
"""

BASE_PATH = "models/checkpoints/Llama3.2-1B-Instruct-int4-qlora-eo8/"
FULLPATH = BASE_PATH + "consolidated.00.pth"

# check if the file exists
if not os.path.exists(FULLPATH):
    raise FileNotFoundError(f"Model file {FULLPATH} not found")

# Load the .pth file
state_dict = torch.load(FULLPATH)

# Save in PyTorch format
torch.save(state_dict, BASE_PATH+ "pytorch_model.bin")

print("Model saved in PyTorch format.")
