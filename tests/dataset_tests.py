import utils

train_dataset = utils.G2NetDataSet(main_folder='/slowfs/datasets/g2net-gravitational-wave-detection',
                                   set_type='train',
                                   labels_file='training_labels.csv')

print(train_dataset[12])
print(next(iter(train_dataset)))

test_dataset = utils.G2NetDataSet(main_folder='/slowfs/datasets/g2net-gravitational-wave-detection',
                                  set_type='test')

print(test_dataset[12])
print(next(iter(test_dataset)))
