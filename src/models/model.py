import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl

class Net(pl.LightningModule):
    def __init__(self, dataset, in_channels, hp, loss_func):
        super(Net, self).__init__()

        self.lr = hp['lr']
        self.dataset = dataset
        self.num_classes = self.dataset.get_num_classes()
        self.loss_func = loss_func

        self.channels = [64]
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.channels[0], 
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

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_func(outputs, labels)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        loss = self.loss_func(output, labels)

        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        _, predicted = torch.max(output,1)
        correct = (predicted == labels).sum()
        total = labels.size(0)

        return {'test_accuracy': round((100*correct/total).item(), 3)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        #self.log("ptl/val_accuracy", avg_acc)
        #return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 
                            lr=self.lr)

