import os
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.mps.is_available():
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device
