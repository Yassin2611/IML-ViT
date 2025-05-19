# wvlet_tree.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class WvletTree(nn.Module):
    """
    Multi-level 2D wavelet decomposition.
    At each level, we get (LL, (LH, HL, HH)).  We collect all of these,
    then upsample each to the size of the first LL.
    Returns a list of length `levels * 4` in this order:
      [LL1, LH1, HL1, HH1,  LL2, LH2, HL2, HH2,  …]
    """
    def __init__(self, in_channels, levels=4, wave='db1'):
        super().__init__()
        self.levels = levels
        self.wave   = wave

    def forward(self, x):
        # x: [B, C, H, W]
        B,C,H,W = x.shape
        bands = []
        curr = x
        for lvl in range(self.levels):
            # we'll build per-band tensors of shape [B,C,h,w]
            LL = torch.zeros(B, C, curr.shape[2]//2, curr.shape[3]//2, device=x.device)
            LH = torch.zeros_like(LL)
            HL = torch.zeros_like(LL)
            HH = torch.zeros_like(LL)
            for b in range(B):
                for c in range(C):
                    arr = curr[b,c].cpu().numpy()
                    ll, (lh, hl, hh) = pywt.dwt2(arr, self.wave)
                    LL[b,c] = torch.from_numpy(ll)
                    LH[b,c] = torch.from_numpy(lh)
                    HL[b,c] = torch.from_numpy(hl)
                    HH[b,c] = torch.from_numpy(hh)
            # store them
            bands.extend([LL, LH, HL, HH])
            curr = LL  # feed next level the low-pass
        # upsample all bands to the spatial size of the first LL
        target_h, target_w = bands[0].shape[2:]
        up = lambda t: F.interpolate(t, size=(target_h,target_w),
                                     mode='bilinear', align_corners=False)
        return [ up(b) for b in bands]

