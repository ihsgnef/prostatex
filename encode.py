import os, pickle
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from wang_args import WangClassifier
import argparse

split = 'valid_bal' # train_bal
# args = argparse.Namespace(mri_sequences='TBAKDEF', data_sequences='TBAKDEF', valid_dir='tbakd3_npy/uint8/5folds/4/train_bal', eval_batch_size=1, pooling=False, embed_dim=10)
args = argparse.Namespace(mri_sequences='ABDEFKT', data_sequences='TBAKDEF', valid_dir='tbakd3_npy/5folds/4/{}'.format(split), eval_batch_size=1, pooling=False, embed_dim=10)
# args = argparse.Namespace(mri_sequences='ABDEFKT', data_sequences='TBAKDEF', valid_dir='tbakd3_npy/5folds/4/{}'.format(split), eval_batch_size=1, pooling=True, embed_dim=10)
# ckpt = '/net/scratch/hanliu/radiology/prostatex/results/wang-fold-bal/z4ktwfw9/checkpoints/epoch=102-valid_acc=0.81.ckpt' # splendid-night-1029
ckpt = '/net/scratch/hanliu/radiology/prostatex/results/wang-emb10/3p0j02wg/checkpoints/epoch=195-valid_auc=0.92.ckpt' # bal_emb10
# ckpt = '/net/scratch/hanliu/radiology/prostatex/results/wang-emb10/1t9z4936/checkpoints/epoch=81-valid_loss=2.15.ckpt' # emb10
model = WangClassifier.load_from_checkpoint(ckpt, **vars(args))
_ = model.eval()
print(model.w_ensemble.numpy())
# w_e = np.array([0.11010821, 0.1750904, 0, 0, 0, 0.5072202 , 0.20758119, 0]) # emb10_bal
# w_e = np.array([0, 0.30813366, 0, 0, 0, 0.31353357, 0, 0.37833276]) # emb10

batches = list(iter(model.val_dataloader()))
inputs = [b[0] for b in batches]
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

path = model.hparams.valid_dir.replace(split, 'embs/{}_findings_emb10.pkl'.format(split))
pickle.dump((fids, inputs, labels, embeds, preds), open(path, "wb"))
print("Encoded {} findings (fids, inputs, labels, embeds, preds) at ".format(split) + path)