import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import model
import utils
import numpy as np

# Define training data loader
train_dataset = utils.G2NetDataSet(main_folder='/slowfs/datasets/g2net-gravitational-wave-detection',
                                   set_type='train',
                                   labels_file='training_labels.csv')
train_loader = DataLoader(train_dataset, batch_size=1)
# Define model
model = model.Model()
model.zero_grad()

# Define loss and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# Training
for epoch in range(50):
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

    torch.save(model, "models/logit_v2_"+str(epoch)+".pt")

print('Finished Training')
torch.save(model, "models/logit_v2.pt")
print('Saved the trained model into logit_v1.pt')
