# Models module for xAR training
# from .diffloss import DiffLoss
# from .flowloss import FlowLoss, SimpleMLPAdaLN
from .flowloss_full import FlowLossFull, FullTransformer
from .jit import jit_base, jit_large, jit_huge, jit_base_32, jit_large_32, jit_huge_32, JiTMAR

# from .mar_ssl import mar_ssl
# from .mar_full import MARFull, mar_full_base, mar_full_large, mar_full_huge
# from .mar_ssl_latent import mar_ssl_latent
# from .mar_original import mar_original_base, mar_original_large, mar_original_huge
from .mar_dino import MAR_DINO
# from .mar_dino_debug import MAR_DINO_DEBUG
from .mar_dino_0416 import MAR_DINO_0416
# from .mar_dino_0416_debug import MAR_DINO_0416_DEBUG
from .mar_dino_0419 import MAR_DINO_0419
from .mar_dino_0419_layerscale import MAR_DINO_0419_LayerScale
from .mar_dino_0420 import MAR_DINO_0420

from .mar import mar_base, mar_large, mar_huge, MAR

