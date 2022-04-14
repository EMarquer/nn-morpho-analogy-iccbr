# %%
from data import Task1Dataset, LANGUAGES
from lepage_sigmorphon import LepageSigmorphonDataset
import torch, torch.nn as nn
import torch.nn.functional as F
from statistics import mean
from classification import Classification
from cnnEmbedding import CNNEmbedding
from functools import partial
import sys
LANGUAGE = sys.argv[1] if len(sys.argv) > 1 else LANGUAGES[0]


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

def pad(tensor, bos_id, eos_id):
    tensor = F.pad(input=tensor, pad=(1,0), mode='constant', value=bos_id)
    tensor = F.pad(input=tensor, pad=(0,1), mode='constant', value=eos_id)
    return tensor

def collate(batch, bos_id, eos_id):
    a_emb, b_emb, c_emb, d_emb = [], [], [], []

    # store the embeddings of each words of every quadruplet
    # beginning = 59, end = 60
    for a, b, c, d in batch:
        #print(a,b,c,d)
        a_emb.append(pad(a, bos_id, eos_id))
        b_emb.append(pad(b, bos_id, eos_id))
        c_emb.append(pad(c, bos_id, eos_id))
        d_emb.append(pad(d, bos_id, eos_id))
        #print("After collate: ", a_emb)

    # make a tensor of all As, af all Bs, of all Cs and of all Ds
    a_emb = torch.stack(a_emb)
    b_emb = torch.stack(b_emb)
    c_emb = torch.stack(c_emb)
    d_emb = torch.stack(d_emb)

    return a_emb, b_emb, c_emb, d_emb

def get_accuracy(y_true, y_prob):
    #print(y_true.size(), y_prob.size(), y_true, y_prob, y_true.ndim, y_prob.ndim)
    assert y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    #print(y_true == y_prob)
    if y_prob.ndim > 1:
        return (y_true == y_prob).sum().item() / y_true.size(0)
    else:
        return (y_true == y_prob).sum().item()



data = torch.load(f"models/classification_CNN_{LANGUAGE}_20e.pth")
train_dataset = Task1Dataset(language=LANGUAGE, mode="train", word_encoding="char")
# Same dict
char_voc_id = voc = train_dataset.char_voc_id
char_voc = train_dataset.char_voc
BOS_ID = len(voc) # (max value + 1) is used for the beginning of sequence value
EOS_ID = len(voc) + 1 # (max value + 2) is used for the end of sequence value

# %%
emb_size = 64
classification_model = Classification(emb_size=16*5) # 16 because 16 filters of each size, 5 because 5 sizes
embedding_model = CNNEmbedding(emb_size=emb_size, voc_size = len(voc) + 2)

classification_model.load_state_dict(data['state_dict_classification'])
embedding_model.load_state_dict(data['state_dict_embeddings'])

dataset = LepageSigmorphonDataset(LANGUAGE)
dataloader = torch.utils.data.DataLoader(dataset, collate_fn = partial(collate, bos_id = BOS_ID, eos_id = EOS_ID))
# %%
positive = 0
negative = 0
for a,b,c,d in dataset:
    a,b,c,d = train_dataset.encode(a,b,c,d)
    # compute the embeddings
    a = embedding_model(a.view(1,-1)).view(1,1,-1)
    b = embedding_model(b.view(1,-1)).view(1,1,-1)
    c = embedding_model(c.view(1,-1)).view(1,1,-1)
    d = embedding_model(d.view(1,-1)).view(1,1,-1)

    is_analogy = torch.squeeze(classification_model(a, b, c, d))
    if (is_analogy).item() > 0.5:
        positive += 1
    else:
        negative += 1

print(f"{LANGUAGE}: {positive/(positive+negative):.2%}")

    

# %%
