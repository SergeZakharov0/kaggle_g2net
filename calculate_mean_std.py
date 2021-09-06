import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from nnAudio.Spectrogram import CQT1992v2
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import wave2spectrogram

main_folder = '/slowfs/datasets/g2net-gravitational-wave-detection'
set_type='train'

train_list = []
for dirpath, dirname, filenames in os.walk(os.path.join(main_folder, set_type)):
    for filename in filenames:
        train_list += [os.path.join(dirpath, filename)]


def yield_data():
    for t in tqdm(train_list):
        yield wave2spectrogram(np.load(t))

def yield_data_sq():
    for t in tqdm(train_list):
        yield torch.pow(wave2spectrogram(np.load(t)), 2)

gen = yield_data()
s = sum(gen)
torch.save(s, 'sum')

gen_2 = yield_data_sq()
ss = sum(gen_2)
torch.save(ss, 'sq_sum')

torch.save(s/len(train_list), 'mean')
torch.save(ss/len(train_list)-torch.pow(s/train_list), 'variance')
torch.save(torch.pow(ss/len(train_list)-torch.pow(s/train_list), 1/2), 'std')