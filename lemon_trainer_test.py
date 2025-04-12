import sys

from mne import epochs
from pandas.core.indexes import multi
from scipy import optimize
sys.path.append("./CBraMod")  # HACK to make imports in submodules work

from lemon_trainer import Trainer
from argparse import Namespace
from lemon_dataset import CustomDataset, LoadDataset
from lemon_model import Model as ModelForLemonGender

params = Namespace(
    data_dir = "data/LEMON_DATA/",
    channels = ['O1', 'O2', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2'],
    segment_size = 512,
    batch_size = 32,
    bandpass_filter = 0.5,
    device = 'mps',

    # model
    foundation_dir = "pretrained_weights/pretrained_weights.pth",
    num_of_classes = 2,

    # finetune
    label_smoothing = 0.1,
    frozen = False,
    optimizer = 'AdamW',
    multi_lr = False,
    lr = 5e-4,
    weight_decay = 5e-2,
    epochs = 10,

)

data_loaders = LoadDataset(params).get_data_loader()
model = ModelForLemonGender(params).to(params.device)
trainer = Trainer(params, data_loaders, model)
trainer.train_for_binaryclass()
