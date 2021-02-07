# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import warnings
warnings.filterwarnings("ignore")

import wandb
wandb.init(
    # mode='offline',
    project="prostatex", 
    group="wang-tbakd3-t3",
    config={
        "loss": "binary_cross_entropy",
        "metric": "accuracy",
        "optimizer": "Adam",
        "lr":1e-4,
        "epoch": 1000,
        "batch_size": 16,
        "augmentation": {
            "degrees": 50,
            "translate": (0.9, 0.9),
            "shear": [-15, 15, -15, 15],
        },
        "train_dir": 'tbakd3_npy/train',
        "valid_dir": 'tbakd3_npy/valid_bal'
        })
wandblogger = WandbLogger()

np.random.seed(0)
torch.manual_seed(0)


## Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(**wandb.config.augmentation)
])

def npy_loader(path: str) -> np.ndarray:
    return np.load(path)

datasets = {}
datasets['train'] = torchvision.datasets.DatasetFolder(wandb.config.train_dir, extensions='npy', loader=npy_loader, transform=transform)
datasets['valid'] = torchvision.datasets.DatasetFolder(wandb.config.valid_dir, extensions='npy', loader=npy_loader, transform=transforms.ToTensor())


class MSDSC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 8, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 5, padding=2, groups=16),
            nn.Conv2d(16, 8, 1)
        )
        self.layer = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat((x1, x2), 1)
        x = self.layer(x)
        return x


class WangClassifier(pl.LightningModule):

    def __init__(self, num_sequences):
        super().__init__()

        self.num_sequences = num_sequences
        self.conv = nn.ModuleList(
            [nn.Sequential(
                MSDSC(), MSDSC(), MSDSC(), MSDSC(), MSDSC(), nn.Flatten()
            ) for i in range(num_sequences)]
        )
        self.linear = nn.ModuleList([nn.Linear(64, 1) for i in range(num_sequences)])
        self.fusion = nn.Linear(64 * num_sequences, 1)

        self.lr = wandb.config.lr
        self.accuracy = lambda x, y: ((x > 0.5).type_as(y) == y).float().mean()
        self.auroc = pl.metrics.functional.classification.auroc

    def forward(self, x):
        conv = [conv(x[:, i].unsqueeze(1).repeat(1, 16, 1, 1)) for i, conv in enumerate(self.conv)]
        conv_x = torch.cat([c.unsqueeze(1) for c in conv], 1)
        linear = [linear(conv_x[:, i]) for i, linear in enumerate(self.linear)]
        linear_x = torch.cat([l.unsqueeze(1) for l in linear], 1)
        fusion_x = self.fusion(torch.cat(conv, 1)).unsqueeze(1)
        x = torch.cat((linear_x, fusion_x), 1)
        x = torch.mean(linear_x, 1)
        return x

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


model = WangClassifier(num_sequences=datasets['train'][0][0].shape[0])
trainer = pl.Trainer(
    logger=wandblogger,
    gpus=-1,
    accelerator='ddp',
    max_epochs=wandb.config.epoch)
trainer.fit(model)