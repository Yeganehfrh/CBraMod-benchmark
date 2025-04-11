import torch
import torch.nn as nn
from models.cbramod import CBraMod

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(32 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, param.num_of_classes)
        )

    def forward(self, x):
        # x = x / 100
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        out = feats.contiguous().view(bz, ch_num*seq_len*200)
        out = self.classifier(out)
        return out