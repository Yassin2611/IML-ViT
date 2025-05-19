# iml_vit_model_wvlet.py
import sys
sys.path.append('./modules')

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# backbone (windowed-ViT)
from modules.window_attention_ViT import ViT as window_attention_vit, LastLevelMaxPool
# new wavelet-tree replacement for SFPN
from modules.wvlet_tree import WvletTree
from modules.decoderhead      import PredictHead

class IMLViT_Wvlet(nn.Module):
    def __init__(
        self,
        # vision transformer backbone args:
        input_size       = 1024,
        patch_size       = 16,
        embed_dim        = 768,
        vit_pretrain_path=None,
        # wavelet-tree args:
        wvlet_levels     = (4, 2, 1, 0.5),  # same scale factors as before
        wavelet          = 'db1',
        # decoder head args:
        mlp_embedding_dim= 256,
        predict_head_norm= 'BN',
        # edge loss:
        edge_lambda      = 20
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size

        # 1) Windowed-ViT encoder
        self.encoder = window_attention_vit(
            img_size = input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[0,1,3,4,6,7,9,10],
            residual_block_indexes=[],
            use_rel_pos=True,
            out_feature="last_feat"
        )
        self.vit_pretrain_path = vit_pretrain_path

        # 2) Replace SFPN with wavelet-tree
        self.feature_extractor = WvletTree(
            in_channels = embed_dim,
            levels      = len(wvlet_levels),
            wave        = wavelet
        )

        # 3) MLP decoder head
        # it will take a list of feature maps (wavelet coeffs)
        self.predict_head = PredictHead(
            feature_channels=[embed_dim for _ in range(len(wvlet_levels)+1)],
            embed_dim=mlp_embedding_dim,
            norm=predict_head_norm
        )

        # losses
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.edge_lambda = edge_lambda

        # init weights
        self.apply(self._init_weights)
        if vit_pretrain_path is not None:
            self._load_mae_weights(vit_pretrain_path)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _load_mae_weights(self, path):
        sd = torch.load(path, map_location='cpu')['model']
        self.encoder.load_state_dict(sd, strict=False)
        print(f"Loaded MAE weights from '{path}'")

    def forward(self, x, masks, edge_masks, shape=None):
        # encode
        feats = self.encoder(x)            # returns dict { 'last_feat': Tensor }
        base  = feats['last_feat']        # [B, C, H', W']

        # wavelet-tree decomposition
        coeffs = self.feature_extractor(base)  # list of [B, C, h_i, w_i]

        # predict head
        pred_logits = self.predict_head(coeffs)

        # upsample to full resolution
        pred = F.interpolate(
            pred_logits,
            size=(self.input_size, self.input_size),
            mode='bilinear',
            align_corners=False
        )

        # compute losses
        loss_main = self.bce_loss(pred, masks)
        loss_edge = F.binary_cross_entropy_with_logits(
            pred, masks, weight=edge_masks
        ) * self.edge_lambda
        loss = loss_main + loss_edge

        return loss, torch.sigmoid(pred), loss_edge
