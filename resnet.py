# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os

from utils.train_test_split import train_test_split
from lesion_extraction_2d.lesion_extractor_2d import get_train_data
from data_visualization.adc_lesion_values import apply_window

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
wandb.init(
    project="prostatex", 
    group="resnet-adc-t11",
    config={
        "loss": "binary_cross_entropy",
        "metric": "accuracy",
        "optimizer": "Adam",
        "lr":0.001,
        "epoch": 1000,
        "batch_size": 32
        })
wandblogger = WandbLogger()

## Data
h5_file_location = os.path.join('./','prostatex-train-ALL.hdf5')
h5_file = h5py.File(h5_file_location, 'r')
train_data_list, train_labels_list, attr = get_train_data(h5_file, ['ADC', 'Ktrans'], size_px=32)
train_data, val_data, train_labels, val_labels = train_test_split(
    train_data_list, train_labels_list, attr, test_size=0.25, random_state=0)

train_data_tensor = torch.Tensor(train_data.astype(np.float32))
train_labels_tensor = torch.Tensor(train_labels)
val_data_tensor = torch.Tensor(val_data.astype(np.float32))
val_labels_tensor = torch.Tensor(val_labels)
all_max = torch.max(train_data_tensor.max(), val_data_tensor.max())
train_data_tensor = train_data_tensor.unsqueeze(dim=1).repeat([1, 3, 1, 1]) / all_max
val_data_tensor = val_data_tensor.unsqueeze(dim=1).repeat([1, 3, 1, 1]) / all_max
pos_weight = 1 / train_labels_tensor.mean()

datasets = {}
datasets["train"] = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)
datasets["val"] = torch.utils.data.TensorDataset(val_data_tensor, val_labels_tensor)

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=wandb.config.batch_size, 
                                              num_workers=16, drop_last=True, shuffle=True)
              for x in ['train', 'val']}

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
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        representations = self.feature_extractor(x)
        return self.classifier(representations)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(y_hat, y.unsqueeze(1))
        acc = self.accuracy(F.sigmoid(y_hat), y)
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(y_hat, y.unsqueeze(1))
        acc = self.accuracy(F.sigmoid(y_hat), y)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self):
        return dataloaders["train"]

    def val_dataloader(self):
        return dataloaders["val"]


model = Resnet18Classifier()
trainer = pl.Trainer(
    logger=wandblogger, 
    gpus=-1, 
    accelerator='ddp',
    max_epochs=wandb.config.epoch)
trainer.fit(model)