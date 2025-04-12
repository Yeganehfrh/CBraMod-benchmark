import torch
import torch.nn as nn
from CBraMod.models.cbramod import CBraMod
from einops.layers.torch import Rearrange

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        self.backbone.load_state_dict(
            torch.load(param.foundation_dir, map_location=torch.device(param.device)))
        self.backbone.proj_out = nn.Identity()

        self.classifier = nn.Sequential(
            Rearrange('b c s p -> b (c s p)'),
            nn.Linear(22*4*200, 4*200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * 200, 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(200, param.num_of_classes),
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        out = self.classifier(feats.contiguous())
        return out
