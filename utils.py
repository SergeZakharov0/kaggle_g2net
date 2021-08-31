import os
import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nnAudio.Spectrogram import CQT1992v2

def preprocess(raw_data):
    # Get CQT transform, create a function for data preprocessing
    transform = CQT1992v2(sr=2048, fmin=22, fmax=1024, 
                        hop_length=32, bins_per_octave=23)
    # Do CQT transform on the signals, merge it as one array
    prep_res = np.concatenate((transform(torch.from_numpy(raw_data[0]).float()), 
                               transform(torch.from_numpy(raw_data[1]).float()), 
                               transform(torch.from_numpy(raw_data[2]).float())))
    # Cut the last time point as it is o-small for our case
    prep_res = prep_res[:,:,0:128]
    # Transpose to be a proper image
    prep_res = prep_res.transpose((1,2,0))
    # Normalize
    prep_res[:,:,0] /= prep_res[:,:,0].max()
    prep_res[:,:,1] /= prep_res[:,:,1].max()
    prep_res[:,:,2] /= prep_res[:,:,2].max()
    return prep_res

def grayFiltering(unfiltered_data, filter_func=np.min):
    # Noise filtering. Rounds everything to the colours of gray
    min_color_data = filter_func(unfiltered_data, axis=2, keepdims=True)
    return np.concatenate((min_color_data, min_color_data, min_color_data), axis=2)
