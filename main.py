import sys
sys.path.append("./CBraMod")  # HACK to make imports in submodules work

import torch
from lemon_model import Model as ModelForLemonGender
from argparse import Namespace

def main():
    print("Hello from cbramod-benchmark!")

    params = {
        "foundation_dir": "pretrained_weights/pretrained_weights.pth",
        "num_of_classes": 2,
        "device": 'mps'
    }

    params = Namespace(**params)
    model = ModelForLemonGender(params).to(params.device)

    # bz, ch_num, seq_len, patch_size = x.shape
    x = torch.randn((8, 22, 4, 200)).to(params.device)
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    main()
