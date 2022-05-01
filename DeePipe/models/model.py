import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()

    def set_architecture(self, architecture):
        self.model = architecture

    def set_hyperparameters(self, optimizer, lr, loss_func):
        self.optimizer = optimizer
        self.lr = lr
        self.loss_func = loss_func
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_func(outputs, labels)

        self.log("ptl/loss", loss)

        return {'loss': loss}

    def training_step_end(self, outputs):
        pass#self.log('ptl/train_loss_batch', outputs['loss'])

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        loss = self.loss_func(output, labels)
        _, predicted = torch.max(output,1)
        correct = (predicted == labels).sum()
        total = labels.size(0)

        self.log("ptl/val_loss", loss)
        self.log("ptl/val_acc", correct/total)

        return {'val_loss': loss, "val_acc": correct/total}

    def validation_step_end(self, outputs):
        self.log("ptl/val_loss_batch", outputs['val_loss'])

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        _, predicted = torch.max(output,1)
        correct = (predicted == labels).sum()
        total = labels.size(0)

        self.log("ptl/test_acc", correct/total)

        return {'test_acc': correct/total}

    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log("ptl/test_acc", avg_acc)
        return avg_acc

    def configure_optimizers(self):
        if self.optimizer.lower() == 'adam':
            return torch.optim.Adam(self.parameters(), 
                            lr=self.lr)

    def predict_step(self, batch, batch_idx):
        # enable Monte Carlo Dropout
        self.dropout = nn.Dropout()
        self.dropout.train()

        # take average of `self.mc_iteration` iterations
        preds = []
        preds = [self.model(item) for item in batch]
        return preds
        
