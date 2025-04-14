
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from src.lemon_dataset import LEMONDataset, LoadDataset
from argparse import Namespace


if __name__ == "__main__":

    params = {
        "data_dir": "data/LEMON_DATA/",
        "channels": ['O1', 'O2', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2'],
        "segment_size": 512,
        "batch_size": 32,
        "bandpass_filter": 0.5,
        "device": 'mps'
    }
    params = Namespace(**params)

    ds = LEMONDataset(
        data_dir=params.data_dir,
        channels=params.channels,
        segment_size=params.segment_size,
        mode='train',
        downstream_task='age',
    )

    subject_ids = ds.subject_ids
    x = ds.x.numpy()
    y = ds.y
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = {'0': class_weights[0], '1': class_weights[1]}

    print("x.shape=", x.shape, "y.shape", y.shape, "subject_ids.shape", subject_ids.shape)
    print("class weights:", class_weights)

    # load via LoadDataset
    data_loaders = LoadDataset(params)
    data_loaders.get_data_loader()

