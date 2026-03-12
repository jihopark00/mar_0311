# Models module for xAR training
from .mar import mar_base, mar_large, mar_huge, MAR
from .diffloss import DiffLoss
from .flowloss import FlowLoss, SimpleMLPAdaLN
from .vae import PatchifyVAE, AutoencoderKL

from .mar_ssl import mar_ssl