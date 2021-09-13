import dataset
import cv2

train_dataset = dataset.SingleLoadDataset(folder='/slowfs/datasets/g2net-224x224/train',
                                          read_function=cv2.imread,
                                          labels_file='/slowfs/datasets/g2net-gravitational-wave-detection/training_labels.csv')

print(train_dataset[12].shape)
print(next(iter(train_dataset)))

