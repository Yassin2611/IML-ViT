# modules/wvlet_head.py
import torch, torch.nn as nn
import torch.nn.functional as F

class WvletHead(nn.Module):
    """
    Simple head that takes all wavelet bands (LL,LH,HL,HH at each level),
    concatenates them, and applies 1×1 fusing + norm + 1×1 predict.
    """
    def __init__(self,
                 in_embed_dim: int,
                 levels: int,
                 predict_channels: int = 1,
                 norm: str = "BN"):
        super().__init__()
        total_ch = in_embed_dim * 4 * levels
        # 1×1 fusion
        self.fuse = nn.Conv2d(total_ch, in_embed_dim, kernel_size=1, bias=False)
        # normalization
        if norm == "BN":
            self.norm = nn.BatchNorm2d(in_embed_dim)
        elif norm == "IN":
            self.norm = nn.InstanceNorm2d(in_embed_dim, affine=True, track_running_stats=True)
        else:
            from modules.decoderhead import LayerNorm
            self.norm = LayerNorm(in_embed_dim)
        # final predictor
        self.pred = nn.Conv2d(in_embed_dim, predict_channels, kernel_size=1)

    def forward(self, bands: list):
        # bands: list of length 4*levels, each [B, C, H, W], already spatially aligned
        x = torch.cat(bands, dim=1)        # [B, total_ch, H, W]
        x = self.fuse(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        return self.pred(x)
