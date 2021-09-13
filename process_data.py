import os
import torch
import cv2
import numpy as np
from nnAudio.Spectrogram import CQT1992v2
from tqdm import tqdm
import argparse

debug = False

# Get CQT transform, create a function for data preprocessing
transform = CQT1992v2(sr=2048, fmin=22, fmax=1024, 
                      hop_length=25, bins_per_octave=30)


def wave2spectrogram(raw_data):
    # Do CQT transform on the signals, merge it as one array
    prep_res = np.concatenate((transform(torch.from_numpy(raw_data[0]).float()),
                               transform(torch.from_numpy(raw_data[1]).float()),
                               transform(torch.from_numpy(raw_data[2]).float())))
    # Transpose to be a proper image
    prep_res = prep_res.transpose((1, 2, 0))
    # Map to [0;1]
    if debug:
        print('wave2spectrogram - max values: ', 
              prep_res[:, :, 0].max(), prep_res[:, :, 1].max(), prep_res[:, :, 2].max())
    prep_res[:, :, 0] = np.rint(prep_res[:, :, 0] / prep_res[:, :, 0].max() * 255)
    prep_res[:, :, 1] = np.rint(prep_res[:, :, 1] / prep_res[:, :, 1].max() * 255)
    prep_res[:, :, 2] = np.rint(prep_res[:, :, 2] / prep_res[:, :, 2].max() * 255)
    return torch.from_numpy(prep_res.astype(np.uint8))


def resize(data, output_size):
    resized_data = cv2.resize(data.numpy(), output_size)
    if debug:
        cv2.imwrite('data.bmp', data.numpy())
        cv2.imwrite('resized_data.bmp', resized_data)
    return resized_data


def process_data(input_folder, output_folder, output_size, input_subset=None):
    if debug:
        print('process_data(', input_folder, output_folder, output_size, input_subset, ')')
    # Getting the list of all entries in folder
    files_list = []
    for dirpath, dirname, filenames in os.walk(input_folder):
        for filename in filenames:
            files_list += [os.path.join(dirpath,filename)]
    # Select subset indexes
    if input_subset is not None:
        files_list = np.array(files_list)[input_subset]
    # Process
    for file_path in tqdm(files_list):
        # Load data
        waves_data = np.load(file_path)
        if debug:
            print('Shape of input data: ', waves_data.shape)
        # Convert to spectrogram with max possible size
        raw_image_data = wave2spectrogram(waves_data)
        if debug:
            print('Shape after CQT: ', raw_image_data.shape)
        # Resize the spectrogram to required shape
        resized_image_data = resize(raw_image_data, output_size)
        if debug:
            print('Shape after resize: ', resized_image_data.shape)
        # Save image
        if debug:
            return
        else:
            cv2.imwrite(os.path.join(output_folder, os.path.basename(file_path)) + '.bmp', resized_image_data)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process waves dataset and convert it to normalized images')
    parser.add_argument('--input', type=str, help='Path to waves dataset')
    parser.add_argument('--output', type=str, help='Path to output image dataset')
    parser.add_argument('--height', type=int, help='Height of output images')
    parser.add_argument('--width', type=int, help='Width of output images')
    parser.add_argument('--first_idx', type=int, default=None, help='First index of processed subset')
    parser.add_argument('--last_idx', type=int, default=None, help='Last index of processed subset')
    args = parser.parse_args()
    input_subset = None
    if args.first_idx is not None and args.last_idx is not None:
        input_subset = range(args.first_idx, args.last_idx)
    
    print('Process the data, convert to images and save')
    process_data(args.input, args.output, (args.height, args.width), input_subset)

    