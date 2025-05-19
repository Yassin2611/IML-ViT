# modules/wvlet_tree.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class WvletTree(nn.Module):
    """
    Multi-level 2D wavelet decomposition.
    At each level, we get (LL, (LH, HL, HH)). We collect all of these,
    then upsample each to the size of the first LL.
    Returns a list of length `levels * 4` in this order:
      [LL1, LH1, HL1, HH1,  LL2, LH2, HL2, HH2,  …]
    """
    def __init__(self, in_channels: int, levels: int = 4, wave: str = 'db1'):
        super().__init__()
        self.levels = levels
        self.wave   = wave

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        bands = []
        curr = x

        for lvl in range(self.levels):
            # allocate output tensors for this level
            h, w = curr.shape[2] // 2, curr.shape[3] // 2
            LL = torch.zeros(B, C, h, w, device=curr.device, dtype=curr.dtype)
            LH = torch.zeros_like(LL)
            HL = torch.zeros_like(LL)
            HH = torch.zeros_like(LL)

            # per-sample, per-channel DWT
            for b in range(B):
                for c in range(C):
                    arr = curr[b, c].cpu().numpy()
                    ll, (lh, hl, hh) = pywt.dwt2(arr, self.wave)
                    LL[b, c] = torch.from_numpy(ll)
                    LH[b, c] = torch.from_numpy(lh)
                    HL[b, c] = torch.from_numpy(hl)
                    HH[b, c] = torch.from_numpy(hh)

            # collect the four bands
            bands.extend([LL, LH, HL, HH])
            # feed the LL band into next level
            curr = LL

        # upsample every band back to the spatial size of the first LL
        target_h, target_w = bands[0].shape[2:]
        def upsample(t: torch.Tensor):
            return F.interpolate(t, size=(target_h, target_w),
                                 mode='bilinear', align_corners=False)

        return [upsample(b) for b in bands]
