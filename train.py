import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import model
import utils
import numpy as np

mean = torch.load('mean')
std = torch.load('std')
# Define training data loader
train_dataset = utils.G2NetDataSet(main_folder='/slowfs/datasets/g2net-gravitational-wave-detection',
                                   set_type='train',
                                   labels_file='training_labels.csv', subset_ind=range(20000),
                                   mean=mean, std=std)
train_loader = DataLoader(train_dataset, batch_size=64)
# Define validation data loader
val_dataset = utils.G2NetDataSet(main_folder='/slowfs/datasets/g2net-gravitational-wave-detection',
                                   set_type='train',
                                   labels_file='training_labels.csv', subset_ind=range(20000, 25000),
                                   mean=mean, std=std)
val_loader = DataLoader(val_dataset, batch_size=64)

# Define models
model = model.Model()
model.zero_grad()

# Define loss and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

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
        outputs = model(inputs)
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
        torch.save(model, "models/logit_v3_"+str(epoch)+".pt")

print('Finished Training')
