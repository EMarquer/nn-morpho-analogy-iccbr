# %%

import pytorch_lightning as pl
from train_reg_univ import Reg, prepare_data
from utils import collate
from functools import partial
import torch

print("\nAttempting load...")
model = Reg.load_from_checkpoint(f"logs/reg_univ/arabic/debug/checkpoints/epoch=0-step=1406-v1.ckpt")
print("Success.")

print("\nAttempting data load...")
train_loader, val_loader, test_loader, voc_size, encode_fn, _ = prepare_data("arabic", 50, 50)
print("Success.")

print("\nAttempting embedding pre-computation...")
model.prepare_embeddings("arabic")
print("Success.")

print("\nAttempting inference...")
for batch in test_loader:
    a,b,c,d = batch
    break

# compute the embeddings
a = model.emb(a)
b = model.emb(b)
c = model.emb(c)
d = model.emb(d)

d_pred = model.reg(a, b, c)

acc_cosine = []
acc_euclid = []
for d__, d_pred_ in zip(d, d_pred):
    #print(d__.size(), d_pred_.size(), self.closest_cosine(d__), self.closest_cosine(d_pred_), self.closest_euclid(d__), self.closest_cosine(d_pred_))
    acc_cosine.append(model.closest_cosine(d__) == model.closest_cosine(d_pred_))
    acc_euclid.append(model.closest_euclid(d__) == model.closest_cosine(d_pred_))

print("acc cosine", acc_cosine)
print("acc euclid", acc_euclid)
print("Success.")
# %%
