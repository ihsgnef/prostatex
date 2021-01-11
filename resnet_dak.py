# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os

from sklearn.model_selection import train_test_split
from lesion_extraction_2d.lesion_extractor_2d import get_train_data
# from data_visualization.adc_lesion_values import apply_window

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
wandb.init(
    project="prostatex", 
    group="resnet-dak-t0",
    config={
        "loss": "binary_cross_entropy",
        "metric": "accuracy",
        "optimizer": "Adam",
        "lr":0.001,
        "epoch": 1000,
        "batch_size": 16
        })
wandblogger = WandbLogger()

## Data
h5_file_location = os.path.join('./','prostatex-train-ALL.hdf5')
h5_file = h5py.File(h5_file_location, 'r')

query = ['BVAL', 'ADC', 'Ktrans']
data = {q: get_train_data(h5_file, [q], size_px=32) for q in query}
findings = {attr['patient_id'] + '-' + attr['fid']:{} for q in query for attr in data[q][2]}
for q in query:
    images, labels, attrs = data[q]
    for i in range(len(images)):
        f = attrs[i]['patient_id'] + '-' + attrs[i]['fid']
        image = images[i].astype(np.float64)
        findings[f][q] = image / image.max()
        if 'label' not in findings[f]:
            findings[f]['label'] = labels[i]
        else:
            assert(findings[f]['label'] == labels[i])
findings = {k:v for k, v in findings.items() if len(v.keys()) == len(query) + 1}
print("Total findings for train/valid: ", len(findings))

images = np.stack([np.stack([findings[f][q] for q in query]) for f in findings])
labels = np.stack([findings[f]['label'] for f in findings])

splitted = train_test_split(
    images, labels, test_size=0.25, random_state=0, stratify=labels)
X_train, X_valid, y_train, y_valid = (torch.Tensor(array) for array in splitted)
pos_weight = 1 / X_valid.mean()

datasets, dataloaders = {}, {}
datasets['train'] = torch.utils.data.TensorDataset(X_train, y_train)
datasets['valid'] = torch.utils.data.TensorDataset(X_valid, y_valid)
dataloaders['train'] = torch.utils.data.DataLoader(
    datasets['train'], batch_size=wandb.config.batch_size, num_workers=16, drop_last=True, shuffle=True)
dataloaders['valid'] = torch.utils.data.DataLoader(
    datasets['valid'], batch_size=wandb.config.batch_size, num_workers=16, drop_last=True)

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
        self.log('valid_loss', loss, sync_dist=True)
        self.log('valid_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self):
        return dataloaders['train']

    def val_dataloader(self):
        return dataloaders['valid']


model = Resnet18Classifier()
trainer = pl.Trainer(
    logger=wandblogger, 
    gpus=[0], 
    # accelerator='ddp',
    max_epochs=wandb.config.epoch)
trainer.fit(model)