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
    # mode="offline",
    project="prostatex", 
    group="resnet-dak-aug-t8",
    config={
        "loss": "binary_cross_entropy",
        "metric": "accuracy",
        "optimizer": "Adam",
        "lr":1e-5,
        "epoch": 1000,
        "batch_size": 16,
        "augmentation": {
            "degrees": 50,
            "translate": (0.1, 0.1),
            "shear": [-15, 15, -15, 15],
        },
        "train_dir": 'dak_images/train',
        "valid_dir": 'dak_images/valid_bal'
        })
wandblogger = WandbLogger()

np.random.seed(0)
torch.manual_seed(0)

## Data
transform = transforms.Compose([
    transforms.RandomAffine(**wandb.config.augmentation),
    transforms.ToTensor()
])

datasets = {}
datasets['train'] = torchvision.datasets.ImageFolder(wandb.config.train_dir, transform=transform)
datasets['valid'] = torchvision.datasets.ImageFolder(wandb.config.valid_dir, transform=transforms.ToTensor())


class Resnet18Classifier(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.layer4 = nn.Identity()
        self.feature_extractor.fc = nn.Identity()
        
        # use the pretrained model to classify
        num_ftrs = 256
        self.classifier = nn.Linear(num_ftrs, 1)
        
        self.lr = wandb.config.lr
        self.accuracy = lambda x, y: ((x > 0.5).type_as(y) == y).float().mean()
        self.auroc = pl.metrics.functional.classification.auroc

    def forward(self, x):
        representations = self.feature_extractor(x)
        return self.classifier(representations)

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


model = Resnet18Classifier()
trainer = pl.Trainer(
    logger=wandblogger, 
    gpus=-1, 
    accelerator='ddp',
    max_epochs=wandb.config.epoch)
trainer.fit(model)