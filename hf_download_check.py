# 

# torch.hub


# rae
from transformers import Dinov2WithRegistersModel
dinov2_path='facebook/dinov2-with-registers-base'
_ =Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)

# mar ssl
import torch
_ = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', pretrained=True)