# iml_vit_model_wvlet.py
import sys
sys.path.append("./modules")

import torch, torch.nn as nn, torch.nn.functional as F
from functools import partial

# 1) ViT + SFPN backbone
from modules.window_attention_ViT import ViT, LastLevelMaxPool

# 2) wavelet-tree feature extractor
from modules.wvlet_tree    import WvletTree

# 3) new, simple wavelet fusion head
from modules.wvlet_head    import WvletHead

class IMLViT_Wvlet(nn.Module):
    def __init__(self,
                 input_size       = 1024,
                 patch_size       = 16,
                 embed_dim        = 768,
                 vit_pretrain_path=None,
                 wvlet_levels     = 4,
                 wavelet          = 'db1',
                 head_norm        = 'BN',
                 edge_lambda      = 20):
        super().__init__()
        self.input_size = input_size

        # (1) windowed‐ViT encoder
        self.encoder = ViT(
            img_size=input_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=12, num_heads=12, drop_path_rate=0.1,
            window_size=14, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[0,1,3,4,6,7,9,10],
            residual_block_indexes=[], use_rel_pos=True,
            out_feature="last_feat"
        )
        if vit_pretrain_path:
            self._load_mae_weights(vit_pretrain_path)

        # (2) wavelet‐tree decomposition
        self.feature_extractor = WvletTree(
            in_channels=embed_dim,
            levels=wvlet_levels,
            wave=wavelet
        )

        # (3) new fusion head
        self.head = WvletHead(
            in_embed_dim=embed_dim,
            levels=wvlet_levels,
            predict_channels=1,
            norm=head_norm
        )

        self.bce_loss   = nn.BCEWithLogitsLoss()
        self.edge_lambda= edge_lambda

    def _load_mae_weights(self, path):
        sd = torch.load(path, map_location="cpu")["model"]
        self.encoder.load_state_dict(sd, strict=False)

    def forward(self, x, masks, edge_masks):
        feats = self.encoder(x)           # dict { 'last_feat': [B,C,h,w] }
        base  = feats['last_feat']

        bands = self.feature_extractor(base)  # list of 4*levels [B,C,hi,wi], each upsampled

        pred_logits = self.head(bands)        # [B,1,h,w]
        pred = F.interpolate(pred_logits,
                             size=(self.input_size, self.input_size),
                             mode="bilinear", align_corners=False)

        loss_main = self.bce_loss(pred, masks)
        loss_edge = F.binary_cross_entropy_with_logits(pred, masks, weight=edge_masks) \
                    * self.edge_lambda
        return loss_main + loss_edge, torch.sigmoid(pred), loss_edge

