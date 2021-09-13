import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import model
import dataset
import numpy as np
import torchvision.models as models
import cv2


def preprocess(x):
    # Transpose for Resnext
    x = x.transpose((2, 1, 0))
    # To [0;1]
    x = x / 255
    # Normalize
    mean = [0.05955338788342238, 0.05964520301283479, 0.07473762879218654]
    std = [0.10204760808674686, 0.10211863743338652, 0.10066736813404807]
    for ch in range(0, 3):
        x[ch] = (x[ch] - mean[ch]) / std[ch]
    return x


labels_file = '/slowfs/datasets/g2net-gravitational-wave-detection/training_labels.csv'
# Define training data loader
train_dataset = dataset.SingleLoadDataset(folder='/slowfs/datasets/g2net-224x224', read_function=cv2.imread,
                                          labels_file=labels_file, subset_ind=range(500000),
                                          preprocess=preprocess)
train_loader = DataLoader(train_dataset, batch_size=64)
# Define validation data loader
val_dataset = dataset.SingleLoadDataset(folder='/slowfs/datasets/g2net-224x224', read_function=cv2.imread,
                                        labels_file=labels_file, subset_ind=range(500000, 560000),
                                        preprocess=preprocess)
val_loader = DataLoader(val_dataset, batch_size=64)

# Define models
model = model.Model(models.resnext50_32x4d())
model.zero_grad()

# Define loss and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training
for epoch in range(1, 50):
    progress_bar = tqdm(train_loader)
    loss_l = []
    for data in progress_bar:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, _ = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs.float()).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_l += [loss.item()]
        # Print statistics
        progress_bar.set_postfix({'loss': np.mean(loss_l)})
    # Validate the epoch
    progress_bar = tqdm(val_loader)

    acc = []
    for data in progress_bar:
        inputs, labels, filename = data
        outputs = model(inputs)
        outputs = (outputs > 0.5).float()
        acc += [(outputs == labels).float().sum()/len(outputs)]
        progress_bar.set_postfix({'acc': float(np.mean(acc))})
    # Save the models after every fifth
    if epoch % 5 == 0:
        torch.save(model, "models/resnext_v1_"+str(epoch)+".pt")

print('Finished Training')
