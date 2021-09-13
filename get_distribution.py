import os
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import warnings


# Calculates distribution of the image dataset
def get_distribution(file_list, image_shape, bins_num=256):
    b = np.zeros((bins_num, image_shape[0], image_shape[1], image_shape[2]))
    for cur_path in tqdm(file_list):
        data = cv2.imread(cur_path)
        if data is None:
            warnings.warn("Warning: image " + cur_path + "cannot be read by cv2!")
            continue
        for h in range(0, image_shape[0]):
            for w in range(0, image_shape[1]):
                for c in range(0, image_shape[2]):
                    v = data[h, w, c]
                    b[v, h, w, c] += 1
    return b


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculate data for distribution and pixel mean')
    parser.add_argument('--input', type=str, help='Path to waves dataset')
    parser.add_argument('--height', type=int, default=224, help='Height of dataset images')
    parser.add_argument('--width', type=int, default=224,  help='Width of dataset images')
    parser.add_argument('--channels', type=int, default=3, help='Channels of dataset images')
    parser.add_argument('--first_idx', type=int, default=None, help='First index of processed subset')
    parser.add_argument('--last_idx', type=int, default=None, help='Last index of processed subset')
    parser.add_argument('--load_bins', type=bool, default=False,
                        help='Load bins.npy instead of calculating it')
    args = parser.parse_args()
    input_subset = None
    if args.first_idx is not None and args.last_idx is not None:
        input_subset = range(args.first_idx, args.last_idx)

    # Get images list
    train_list = []
    for dirpath, dirname, filenames in os.walk(args.input):
        for filename in filenames:
            train_list += [os.path.join(dirpath, filename)]
    train_list = np.array(train_list)[input_subset]

    # Calculate distribution or load it from file
    if not args.load_bins:
        bins = get_distribution(train_list, (args.height, args.width, args.channels))
        np.save('bins.npy', bins)
    else:
        bins = np.load('bins.npy')

    # Calculate pixel mean and display
    pixel_mean = [0] * args.channels
    ch_std = [0] * args.channels
    for ch in range(0, args.channels):
        pixel_sum = 0
        pixel_num = 0
        squares_sum = 0
        for vl in range(1, 256):
            pixel_sum += np.sum(bins[vl, :, :, ch]) * (vl / 255)
            squares_sum += np.sum(bins[vl, :, :, ch]) * (vl / 255) * (vl / 255)
            pixel_num += np.sum(bins[vl, :, :, ch])
        pixel_mean[ch] = pixel_sum / pixel_num
        ch_std[ch] = np.sqrt(squares_sum / pixel_num - pixel_mean[ch] * pixel_mean[ch])
    print(pixel_mean)
    print(ch_std)
