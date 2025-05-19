import torch, torch.nn as nn, torch.nn.functional as F
import pywt

class WvletTree(nn.Module):
    def __init__(self, in_channels, levels=4, wave='db1'):
        super().__init__()
        self.levels = levels
        self.wave   = wave

    def forward(self, x):
        # x: [B,C,H,W]
        feats, curr = [], x
        for _ in range(self.levels):
            B,C,H,W = curr.shape
            ll_batches = []
            for b in range(B):
                ch_feats = []
                for c in range(C):
                    # numpy DWT then back to torch
                    arr = curr[b,c].cpu().numpy()
                    ll, _ = pywt.dwt2(arr, self.wave)
                    ch_feats.append(torch.from_numpy(ll).to(curr.device))
                ll_batches.append(torch.stack(ch_feats,0))
            curr = torch.stack(ll_batches,0)     # [B,C,H/2,W/2]
            feats.append(curr)
        # upsample all to feats[0].shape[2:]
        target = feats[0].shape[2:]
        return [
            F.interpolate(f, size=target, mode='bilinear', align_corners=False)
            for f in feats
        ]
