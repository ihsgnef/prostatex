# -*- coding: utf-8 -*-
import os
import time
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc, stat_scores
from pytorch_lightning.loggers import WandbLogger
import wandb

import warnings
warnings.filterwarnings("ignore")


class MSDSC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1)
        )
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels * 2),
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

    def __init__(self, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.num_sequences = self.hparams.num_sequences
        # self.fn_penalty = self.hparams.fn_penalty
        self.conv = nn.ModuleList(
            [nn.Sequential(
                *([MSDSC(16*2**i) for i in range(5)] + [nn.Flatten()])
            ) for i in range(self.num_sequences)]
        )
        self.linear = nn.ModuleList([nn.Linear(16*2**5*2*2, 1) for i in range(self.num_sequences)])
        self.fusion = nn.Linear(16*2**5*2*2 * self.num_sequences, 1)

    def metrics(self, prob, target, threshold=0.5):
        pred = (prob >= threshold).long()
        tp, fp, tn, fn, sup = stat_scores(pred, target, class_index=1)
        m = {}
        m['auc'] = auroc(prob, target) if 0 < sup < len(target) else None
        m['acc'] = (tp + tn) / (tp + tn + fp + fn)
        m['tpr'] = tp / (tp + fn)
        m['tnr'] = tn / (tn + fp)
        m['ppv'] = tp / (tp + fp)
        m['f1'] = 2 * tp / (2 * tp + fp + fn)
        return m

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
        y_hat = torch.sigmoid(self(x)).view(-1)
        criterion = nn.BCELoss()
        loss = criterion(y_hat, y.float())
        m = self.metrics(y_hat, y)
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', m['acc'], prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.sigmoid(self(x)).view(-1)
        criterion = nn.BCELoss()
        loss = criterion(y_hat, y.float())
        m = self.metrics(y_hat, y)
        self.log('valid_loss', loss, sync_dist=True)
        self.log('valid_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('valid_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('valid_sensitivity', m['tpr'], sync_dist=True)
        self.log('valid_specificity', m['tnr'], sync_dist=True)
        self.log('valid_precision', m['ppv'], sync_dist=True)
        self.log('valid_f1', m['f1'], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
    def parse_augmentation(self):
        affine = {}
        affine["degrees"] = self.hparams.rotate
        if self.hparams.translate > 0: 
            translate = self.hparams.translate
            affine["translate"] = (translate, translate)
        if self.hparams.scale > 0: 
            scale = self.hparams.scale
            affine["scale"] = (1 - scale, 1 + scale)
        if self.hparams.shear > 0:
            shear = self.hparams.shear
            affine["shear"] = (-shear, shear, -shear, shear)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(self.hparams.horizontal_flip),
            transforms.RandomVerticalFlip(self.hparams.vertical_flip),
            transforms.RandomAffine(**affine)
        ])
        return transform

    def train_dataloader(self):
        dataset = torchvision.datasets.DatasetFolder(
            self.hparams.train_dir, extensions='npy', loader=np.load, transform=self.parse_augmentation()
            )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.hparams.train_batch_size, 
            num_workers=self.hparams.dataloader_num_workers, 
            drop_last=True, shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataset = torchvision.datasets.DatasetFolder(
            self.hparams.valid_dir, extensions='npy', loader=np.load, transform=transforms.ToTensor()
            )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.hparams.eval_batch_size, 
            num_workers=self.hparams.dataloader_num_workers, 
            drop_last=False, shuffle=False)
        return dataloader

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--num_sequences", default=7, type=int, help="Number of MRI sequences")
        parser.add_argument("--fn_penalty", default=20, type=int, help="Penalty for false negatives")
        parser.add_argument("--horizontal_flip", default=0, type=float)
        parser.add_argument("--vertical_flip", default=0, type=float)
        parser.add_argument("--rotate", default=0, type=int)
        parser.add_argument("--translate", default=0, type=float)
        parser.add_argument("--scale", default=0, type=float)
        parser.add_argument("--shear", default=0, type=float)
        return parser

def train(
    model:WangClassifier,
    args: argparse.Namespace,
    early_stopping_callback=False,
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs
    ):

    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.path.join(model.hparams.output_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    # build logger
    ## WandB logger
    experiment = wandb.init(
        mode=args.wandb_mode, 
        group=args.wandb_group
    )
    logger = WandbLogger(
        project="prostatex",
        experiment=experiment
    )

    # add custom checkpoints
    ckpt_path = os.path.join(
        args.output_dir, logger.version, "checkpoints",
    )
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{valid_auc:.2f}", monitor="valid_auc", mode="max", save_top_k=1, verbose=True
        )

    train_params = {}
    train_params["max_epochs"] = args.max_epochs
    if args.gpus == -1 or args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        callbacks=extra_callbacks,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model)
        # save best model to `best_model.ckpt`
        target_path = os.path.join(ckpt_path, 'best_model.ckpt')
        logger.info(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
        shutil.copy(checkpoint_callback.best_model_path, target_path)

    return trainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--loss", default="binary_cross_entropy", type=str)
    parser.add_argument("--metric", default="accuracy", type=str)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--dataloader_num_workers", default=16, type=int)
    parser.add_argument("--train_dir", default=None, type=str, required=True)
    parser.add_argument("--valid_dir", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--wandb_mode", default="online", type=str)
    parser.add_argument("--do_train", action="store_true")
    WangClassifier.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{__name__}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)
    
    dict_args = vars(args)
    model = WangClassifier(**dict_args)
    trainer = train(model, args)

if __name__ == "__main__":
    main()