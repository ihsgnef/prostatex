# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os

from sklearn.model_selection import train_test_split
from lesion_extraction_2d.lesion_extractor_2d import get_train_data

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
wandb.init(
    project="prostatex", 
    group="xmas-dak-cat-t2",
    config={
        "loss": "binary_cross_entropy",
        "metric": "accuracy",
        "optimizer": "Adam",
        "lr":1e-5,
        "epoch": 1000,
        "batch_size": 16
        })
wandblogger = WandbLogger()

np.random.seed(0)
torch.manual_seed(0)

## Data
transform = transforms.Compose([
    transforms.RandomAffine(degrees=50, translate=(0.9, 0.9), shear=[-15, 15, -15, 15]),
    transforms.ToTensor()
])

datasets = {}
datasets['train'] = torchvision.datasets.ImageFolder('dak_images/train', transform=transform)
datasets['valid'] = torchvision.datasets.ImageFolder('dak_images/valid', transform=transforms.ToTensor())


class XmasNetClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256)
        )
        self.classifier = nn.Linear(256 * 3, 1)

        self.lr = wandb.config.lr
        self.accuracy = lambda x, y: ((x > 0.5).type_as(y) == y).float().mean()
        self.auroc = pl.metrics.functional.classification.auroc

    def forward(self, x):
        x1 = self.conv1(torch.unsqueeze(x[:,0], 1))
        x2 = self.conv2(torch.unsqueeze(x[:,1], 1))
        x3 = self.conv3(torch.unsqueeze(x[:,2], 1))
        x = torch.cat([x1, x2, x3], dim=1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.type_as(y_hat).unsqueeze(1)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat, y)
        acc = self.accuracy(torch.sigmoid(y_hat), y)
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.type_as(y_hat).unsqueeze(1)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat, y)
        acc = self.accuracy(torch.sigmoid(y_hat), y)
        auc = self.auroc(torch.sigmoid(y_hat).squeeze(), y.squeeze())
        self.log('valid_loss', loss, sync_dist=True)
        self.log('valid_acc', acc, prog_bar=True, sync_dist=True)
        self.log('valid_auc', auc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            datasets['train'], 
            batch_size=wandb.config.batch_size, 
            num_workers=16, drop_last=True, shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            datasets['valid'], 
            batch_size=64, 
            num_workers=16, drop_last=False)
        return dataloader


model = XmasNetClassifier()
trainer = pl.Trainer(
    logger=wandblogger, 
    gpus=-1, 
    accelerator='ddp',
    max_epochs=wandb.config.epoch)
trainer.fit(model)