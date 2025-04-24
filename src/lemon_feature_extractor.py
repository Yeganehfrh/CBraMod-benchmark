import sys

from lemon_dataset import LEMONDataset
sys.path.append("./CBraMod")  # HACK to make imports in submodules work

import torch
from torch import nn
from CBraMod.models.cbramod import CBraMod
from einops.layers.torch import Rearrange
from argparse import Namespace

DEFAULT_PARAMS = Namespace(**{
    "foundation_dir": "pretrained_weights/pretrained_weights.pth",
    "features_file_path": "data/LEMON/CBraMod_features_<DOWNSTREAM_TASK>.pt",
    "num_of_classes": 2,
    "device": 'cpu',

    "data_dir": "data/LEMON/",
    "channels": ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3',
                 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'],
    "downstream_task": "gender",
    "segment_size": 30,  # TODO
    "batch_size": 1024,
    "bandpass_filter": 0.3,
    "n_channels": 19,
    "n_segments": 2,  # TODO ?

})

DEFAULT_PARAMS.features_file_path = DEFAULT_PARAMS.features_file_path.replace(
    "<DOWNSTREAM_TASK>", DEFAULT_PARAMS.downstream_task.lower())


def extract_and_save_features(params):

    backbone = CBraMod(
        in_dim=200, out_dim=200, d_model=200,
        dim_feedforward=800, seq_len=30,
        n_layer=12, nhead=8
    )

    backbone.load_state_dict(
        torch.load(params.foundation_dir,
                   map_location=torch.device(params.device)))
    backbone.proj_out = nn.Identity()

    ds = LEMONDataset(
        data_dir=params.data_dir,
        channels=params.channels,
        downstream_task=params.downstream_task,
        segment_size=params.segment_size,
        mode='all')
    x = ds.x.to(params.device)

    # bz, ch_num, seq_len, patch_size = x.shape
    features = backbone(x)
    features = Rearrange('b c s p -> b (c s p)')(features).contiguous()
    features = features.flatten(1)
    print("features.shape=", features.shape)

    # store features
    torch.save({'features': features.detach().numpy(),
                'gender': ds.y,
                'subject_ids': ds.subject_ids}, params.features_file_path)
    print("features saved to", params.features_file_path)


if __name__ == "__main__":

    extract_and_save_features(DEFAULT_PARAMS)

    # Test the stored file
    print("[TEST] Loading features...")
    feat_ds = torch.load(DEFAULT_PARAMS.features_file_path, weights_only=False)
    print("[TEST] features.shape =", feat_ds['features'].shape)
    print("[TEST] gender.shape =", feat_ds['gender'].shape)
    print("[TEST] subject_ids.shape =", len(feat_ds['subject_ids']))
    print("[TEST] Done.")
