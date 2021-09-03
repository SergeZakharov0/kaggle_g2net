import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from nnAudio.Spectrogram import CQT1992v2

# Get CQT transform, create a function for data preprocessing
transform = CQT1992v2(sr=2048, fmin=22, fmax=1024,
                      hop_length=32, bins_per_octave=23)


def wave2spectrogram(raw_data):
    # Do CQT transform on the signals, merge it as one array
    prep_res = np.concatenate((transform(torch.from_numpy(raw_data[0]).float()),
                               transform(torch.from_numpy(raw_data[1]).float()),
                               transform(torch.from_numpy(raw_data[2]).float())))
    # Cut the last time point as it is o-small for our case
    prep_res = prep_res[:, :, 0:128]
    # Transpose to be a proper image
    prep_res = prep_res.transpose((1, 2, 0))
    # Normalize
    prep_res[:, :, 0] /= prep_res[:, :, 0].max()
    prep_res[:, :, 1] /= prep_res[:, :, 1].max()
    prep_res[:, :, 2] /= prep_res[:, :, 2].max()
    return prep_res


def gray_filtering(unfiltered_data, filter_func=np.min):
    # Noise filtering. Rounds everything to the colours of gray
    min_color_data = filter_func(unfiltered_data, axis=2, keepdims=True)
    return np.concatenate((min_color_data, min_color_data, min_color_data), axis=2)


class G2NetDataSet(Dataset):
    def __init__(self, main_folder, set_type, labels_file=None):
        Dataset.__init__(self)
        self.main_folder = main_folder
        self.set_type = set_type
        # Getting the list of all entries in folder
        self.files_list = []
        for dirpath, dirname, filenames in os.walk(os.path.join(main_folder, set_type)):
            for filename in filenames:
                self.files_list += [os.path.join(dirpath,filename)]
        # Read labels if necessary
        if labels_file:
            self.labels_list = pd.read_csv(os.path.join(main_folder, labels_file))
        else:
            self.labels_list = None

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        filename_short = os.path.basename(self.files_list[idx]).split('.')[0]
        # Add labels if possible
        if self.labels_list is not None:
            label = self.labels_list[self.labels_list.id == filename_short].to_numpy()[0, 1]
        else:
            label = None

        return wave2spectrogram(np.load(self.files_list[idx])), np.array([label], dtype=float), filename_short