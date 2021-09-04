import torch


# Try simple logit for now
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(128 * 128 * 3, 128 * 128)
        self.fc3 = torch.nn.Linear(128 * 128, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.sigmoid(self.fc3(x))
        return x.double()
