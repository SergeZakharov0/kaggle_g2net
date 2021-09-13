import torch
import torchvision.models as models


class Model(torch.nn.Module):
    def __init__(self, cnn_model):
        super(Model, self).__init__()
        self.cnn = cnn_model
        self.tail = torch.nn.Sequential(torch.nn.Linear(1000, 1))

    def forward(self, x):
        x = self.cnn(x)
        x = torch.sigmoid(self.tail(x))
        return x.double()
