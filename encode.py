import os, pickle
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from wang_args import WangClassifier
import argparse

args = argparse.Namespace(mri_sequences='TBAKDEF', data_sequences='TBAKDEF', valid_dir='tbakd3_npy/uint8/5folds/4/train_bal', eval_batch_size=1, pooling=True)
ckpt = '/net/scratch/hanliu/radiology/prostatex/results/wang-fold-bal/z4ktwfw9/checkpoints/epoch=102-valid_acc=0.81.ckpt' # splendid-night-1029
model = WangClassifier.load_from_checkpoint(ckpt, **vars(args))
_ = model.eval()

batches = list(iter(model.valid_dataloader()))
inputs = [b[0][:, model.mri_index] for b in batches]
labels = [b[1] for b in batches]
fids = sorted(os.listdir(model.hparams.valid_dir+'/0')) + sorted(os.listdir(model.hparams.valid_dir+'/1'))
fids = [fid.replace('.npy', '') for fid in fids]

embeds = [model.embed(im) for im in inputs]
preds = [model.predict(im, multi=True) for im in inputs]

fids = np.asarray(fids)
inputs = np.asarray([i.squeeze().detach().numpy() for i in inputs])
labels = np.asarray([l.squeeze().detach().numpy() for l in labels])
embeds = np.asarray([e.squeeze().detach().numpy() for e in embeds])
preds = np.asarray([p.squeeze().detach().numpy() for p in preds])

path = model.hparams.valid_dir.replace('train_bal', 'train_findings.pkl')
pickle.dump((fids, inputs, labels, embeds, preds), open(path, "wb"))
print("Encoded training findings (fids, inputs, labels, embeds, preds) at " + path)