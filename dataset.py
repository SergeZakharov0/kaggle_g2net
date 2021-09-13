import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class G2NetDataset(Dataset):
    def __init__(self, folder, preprocess=(lambda x: x), read_function=torch.load,
                 subset_ind=None, labels_file=None):
        Dataset.__init__(self)
        self.preprocess = preprocess
        self.read_function = read_function
        # Getting the list of all entries in folder
        self.files_list = []
        for dirpath, dirname, filenames in os.walk(folder):
            for filename in filenames:
                self.files_list += [os.path.join(dirpath, filename)]
        # Select subset indexes
        if subset_ind is not None:
            self.files_list = np.array(self.files_list)[subset_ind]
        # Read labels if necessary
        if labels_file:
            self.labels_list = pd.read_csv(labels_file)
            self.labels_list.index = self.labels_list.id.values
        else:
            self.labels_list = None

    def __len__(self):
        return len(self.files_list) 


class SingleLoadDataset(G2NetDataset):
    def __init__(self, folder, preprocess=(lambda x: x), read_function=torch.load,
                 subset_ind=None, labels_file=None):
        G2NetDataset.__init__(self, folder, preprocess, read_function, subset_ind, labels_file)

    def __getitem__(self, idx):
        shortname = (os.path.basename(self.files_list[idx])).split('.')[0]
        # Add labels if possible
        if self.labels_list is not None:
            label = self.labels_list.loc[shortname].to_numpy()[1]
        else:
            label = None
        return self.preprocess(self.read_function(self.files_list[idx])), float(label), shortname


class SimplePagedDataset(G2NetDataset):
    def __init__(self, folder, preprocess=(lambda x: x), read_function=torch.load,
                 subset_ind=None, labels_file=None, page_size=1):
        G2NetDataset.__init__(self, folder, preprocess, read_function, subset_ind, labels_file)
        # Preload first page
        self.page_idx = 0
        self.page_size = page_size
        self.page_data = [self.preprocess(self.read_function(x))
                          for x in self.files_list[0:self.page_size]]

    def __getitem__(self, idx):
        shortname = (os.path.basename(self.files_list[idx])).split('.')[0]
        # Add labels if possible
        if self.labels_list is not None:
            label = self.labels_list.loc[shortname].to_numpy()[1]
        else:
            label = None
        # Load required page
        page_idx = int(idx / self.page_size)
        if self.page_idx != page_idx:
            self.page_data = [self.preprocess(self.read_function(x))
                              for x in self.files_list[(page_idx*self.page_size):
                                                       ((page_idx+1)*self.page_size)]]
            self.page_idx = page_idx
        return self.page_data[idx % self.page_size], float(label), shortname
