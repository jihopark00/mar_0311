# Models module for xAR training
from .mar import mar_base, mar_large, mar_huge, MAR
from .diffloss import DiffLoss
from .flowloss import FlowLoss, SimpleMLPAdaLN
from .flowloss_full import FlowLossFull, FullTransformer
from .vae import PatchifyVAE, AutoencoderKL
from .jit import jit_base, jit_large, jit_huge, jit_base_32, jit_large_32, jit_huge_32, JiTMAR

from .mar_ssl import mar_ssl
from .mar_full import MARFull, mar_full_base, mar_full_large, mar_full_huge
from .mar_ssl_latent import mar_ssl_latent