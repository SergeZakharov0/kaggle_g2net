import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import model
import utils

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
for epoch in range(1):
    progress_bar = tqdm(train_loader)
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

        # Print statistics
        progress_bar.set_postfix({'loss': loss.item()})

print('Finished Training')
torch.save(model, "models/logit_v1.pt")
print('Saved the trained model into logit_v1.pt')