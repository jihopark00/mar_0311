import torch.nn as nn

from models.mar_dino import MAR_DINO


class MAR_DINO_DEBUG(MAR_DINO):
    """Debug variant of MAR_DINO.

    Adds `replace_ls_with_identity`: when True, every dinov2 block's `ls1`
    and `ls2` (LayerScale) modules are swapped for nn.Identity, removing
    LayerScale from the residual path entirely.
    """

    def __init__(self, *args, replace_ls_with_identity=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.replace_ls_with_identity = replace_ls_with_identity
        if replace_ls_with_identity:
            for blk in self.dinov2_backbone.blocks:
                blk.ls1 = nn.Identity()
                blk.ls2 = nn.Identity()
