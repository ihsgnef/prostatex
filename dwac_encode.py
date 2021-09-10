import os, pickle
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from dwac_args import DWAC
import argparse

args = argparse.Namespace(mri_sequences='ABDEFKT', data_sequences='TBAKDEF', train_dir='tbakd3_npy/5folds/4/train', valid_dir='tbakd3_npy/5folds/4/valid', embed_dim=10, pooling=True, merge_seq=True, merge_dim=80)

# ckpt = '/net/scratch/hanliu/radiology/prostatex/results/dwac-emb10/3922buuj/checkpoints/epoch=62-valid_loss=0.38.ckpt' # emb10
# ckpt = '/net/scratch/hanliu/radiology/prostatex/results/dwac-emb10-merge/1zzpiwyt/checkpoints/epoch=46-valid_loss=0.36.ckpt' # emb10.merged
ckpt = '/net/scratch/hanliu/radiology/prostatex/results/dwac-emb10-merge80/31n9n4uw/checkpoints/epoch=32-valid_loss=0.39.ckpt' # emb10.merged80
model = DWAC.load_from_checkpoint(ckpt, **vars(args))
model.eval()
torch.no_grad()

batches = list(iter(model.val_dataloader()))
inputs = batches[0][0]
labels = batches[0][1]
embeds = model.embed(inputs)

batch = list(iter(model.ref_dataloader()))
ref_x = batch[0][0]
ref_y = batch[0][1]
ref_z = model.embed(ref_x)

val_fids = sorted(os.listdir(model.hparams.valid_dir+'/0')) + sorted(os.listdir(model.hparams.valid_dir+'/1'))
val_fids = [fid.replace('.npy', '') for fid in val_fids]

ref_fids = sorted(os.listdir(model.hparams.train_dir+'/0')) + sorted(os.listdir(model.hparams.train_dir+'/1'))
ref_fids = [fid.replace('.npy', '') for fid in ref_fids]

val_fids = np.asarray(val_fids)
inputs = np.asarray([i.squeeze().detach().numpy() for i in inputs])
labels = np.asarray([l.squeeze().detach().numpy() for l in labels])
embeds = np.asarray([e.squeeze().detach().numpy() for e in embeds])

ref_fids = np.asarray(ref_fids)
ref_x = np.asarray([i.squeeze().detach().numpy() for i in ref_x])
ref_y = np.asarray([l.squeeze().detach().numpy() for l in ref_y])
ref_z = np.asarray([e.squeeze().detach().numpy() for e in ref_z])

path = model.hparams.valid_dir.replace('valid', 'embs/dwac_valid_emb10.merged80.pkl')
pickle.dump((val_fids, inputs, labels, embeds), open(path, "wb"))
print("Encoded valid embeddings (fids, inputs, labels, embeds) at " + path)

path = model.hparams.train_dir.replace('train', 'embs/dwac_train_emb10.merged80.pkl')
pickle.dump((ref_fids, ref_x, ref_y, ref_z), open(path, "wb"))
print("Encoded train embeddings (fids, inputs, labels, embeds) at " + path)