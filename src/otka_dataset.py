import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import xarray as xr
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

from scipy.signal import resample
import numpy as np
import xarray as xr
import pandas as pd
import torch
from scipy.signal import resample, butter, sosfiltfilt
from sklearn.preprocessing import RobustScaler, StandardScaler

def preprocess_data(data, baseline_duration=0.5, sampling_rate=128):
    # Step 1: Baseline correction (subtract the mean of the first 0.5 seconds for each channel)
    sample_size = data.shape[0]
    baseline_samples = int(baseline_duration * sampling_rate)
    baseline_mean = np.mean(data[:, :, :baseline_samples], axis=-1, keepdims=True)
    data_corrected = data - baseline_mean
    # print_stats(data_corrected, 'Corrected')

    # Step 2: Normalize using median and IQR
    scaler = RobustScaler()
    data_corrected = data_corrected.transpose(0, 2, 1)  # Transpose for sklearn (samples, times, features)
    data_scaled = np.array([scaler.fit_transform(data_corrected[i]) for i in range(sample_size)])
    # print_stats(data_scaled, 'Scaled')

    # Step 3: Z-score normalization
    normalizer = StandardScaler()
    data_normalized = np.array([normalizer.fit_transform(data_scaled[i]) for i in range(sample_size)]).transpose(0, 2, 1)
    # print_stats(data_normalized, 'Normalized')

    # Step 4: Clamp values greater than 20 standard deviations
    std_threshold = 20
    data_clamped = np.clip(data_normalized, -std_threshold, std_threshold)
    # print_stats(data_clamped, 'Clamped')

    return data_clamped


def format_subject_id(subject_id):
    return f"sub-{int(subject_id):02d}"


def load_data(eeg_data_path, behavioral_path, channels,
                   time_dim=512, downsample=True):
    EEG = xr.open_dataarray(eeg_data_path, engine='h5netcdf')
    behavioral = pd.read_csv(behavioral_path)

    classes = behavioral[['gender', 'bids_id']].dropna().set_index('bids_id')
    classes['gender'] = classes['gender'].apply(lambda x: 0 if x == 'Male' else 1)

    if downsample:
        n_y0 = (classes == 0).sum().values
        n_y1 = (classes == 1).sum().values
        n_min = min(n_y0, n_y1)
        n_subjects = n_min * 2
        y0_sub_ids = classes.query("gender == 0").index[:n_min[0]]
        y1_sub_ids = classes.query("gender == 1").index[:n_min[0]]
        sub_ids = y0_sub_ids.append(y1_sub_ids)
        sub_ids_formatted = [format_subject_id(sub_id) for sub_id in sub_ids]
        EEG = EEG.sel(subject=sub_ids_formatted)

    # X_input
    x = EEG.sel(channel=channels).to_numpy()

    # resample and preprocess
    n_samples = int((x.shape[-1] / 128) * 98)
    x = resample(x, num=n_samples, axis=-1)
    x = x.reshape(-1, *x.shape[2:])
    x = preprocess_data(x, sampling_rate=98)
    # lowpass filter
    sos = butter(4, 0.5, btype='high', fs=98, output='sos')
    x = sosfiltfilt(sos, x, axis=-1)
    x = torch.tensor(x.copy()).unfold(2, time_dim, time_dim).permute(0, 2, 3, 1).flatten(0, 1)

    # Classes
    n_subjects = len(sub_ids)
    y = classes.loc[sub_ids].values
    y = y.repeat(x.shape[0] // n_subjects)

    # Groups
    sub = torch.tensor(np.arange(0, n_subjects).repeat(x.shape[0] // n_subjects)[:, np.newaxis])
    groups = sub.squeeze().numpy()

    return x, y, groups


class OTKADataset(Dataset):
    def __init__(
            self,
            data_dir,
            channels=['O1', 'O2', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2'],
            downstream_task='age',
            segment_size: int = 512,
            mode='train',
    ):
        super(OTKADataset, self).__init__()
        self.mode = mode
        self.channels = channels

        x, y, groups = load_data(
            Path(data_dir) / "eeg.nc5",
            Path(data_dir) / "beh.csv",
            ['O1', 'O2', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2'], downsample=True)
        x = x.permute(0, 2, 1)  # bz, ch_num, seq_len

        self.n_subjects = len(np.unique(groups))

        # segment signals
        x = x.unfold(2, segment_size, segment_size).permute(0, 2, 3, 1).flatten(0, 1)  
        # x.shape: bz, seq_len, ch_num
        # TODO expected: bz, ch_num, seq_len, patch_size
        patch_size = 200

        self.subject_ids = torch.tensor(np.arange(0, self.n_subjects).repeat(x.shape[0] // self.n_subjects)[:, np.newaxis])
        self.subject_ids = self.subject_ids.squeeze()

        x = x.permute(0, 2, 1)
        x = x.unfold(2, patch_size, patch_size)
        self.x = x.float()

        self.y = y

        # TODO test/train split
        indices = torch.randperm(self.y.shape[0])
        self.x=self.x[indices]
        self.y=self.y[indices]

        if self.mode == 'train':
            self.x = self.x[:int(len(self.x) * 0.8)]
            self.y = self.y[:int(len(self.y) * 0.8)]
            self.subject_ids = self.subject_ids[:int(len(self.subject_ids) * 0.8)]
        elif self.mode == 'val':
            self.x = self.x[int(len(self.x) * 0.8):int(len(self.x) * 0.9)]
            self.y = self.y[int(len(self.y) * 0.8):int(len(self.y) * 0.9)]
            self.subject_ids = self.subject_ids[int(len(self.subject_ids) * 0.8):int(len(self.subject_ids) * 0.9)]
        elif self.mode == 'test':
            self.x = self.x[int(len(self.x) * 0.9):int(len(self.x) * 1.)]
            self.y = self.y[int(len(self.y) * 0.9):int(len(self.y) * 1.)]
            self.subject_ids = self.subject_ids[int(len(self.subject_ids) * 0.9):int(len(self.subject_ids) * 1.)]


    def __len__(self):
        return len((self.y))

    def __getitem__(self, idx):
        x_idx = self.x[idx]
        y_idx = self.y[idx]
        return x_idx, y_idx

    def collate(self, batch):
        x = np.array([x[0] for x in batch])
        y = np.array([x[1] for x in batch])
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.data_dir = params.data_dir
        self.channels = params.channels
        self.segment_size = params.segment_size

    def get_data_loader(self):
        train_set = OTKADataset(self.data_dir, mode='train')
        val_set = OTKADataset(self.data_dir, mode='val')
        test_set = OTKADataset(self.data_dir, mode='test')
        print("train,val,test: ", len(train_set), len(val_set), len(test_set))
        print("total: ", len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=True,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=True,
            ),
        }
        return data_loader
