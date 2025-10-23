import torch

class ClassificationImageModel(torch.nn.Module):

    def __init__(self):
        super(ClassificationImageModel, self).__init__()
            
        self.conv1 = torch.nn.Conv2d(1, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 6, 3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(5 * 5 * 6, 128)
        self.linear2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x