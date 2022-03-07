import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.num_classes = num_classes
        self.channels = [64]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.channels[0], 
            kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=self.channels[0], out_channels=32, 
            kernel_size=4, stride=1, padding=2)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.relu = nn.ReLU()

        self.linear = nn.Linear(7200, self.num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x