import os
from config import RETRIEVAL_TMP, RETRIEVAL_CACHE, RETRIEVAL_VECTORS_PATH, check_dir
import torchtext.vocab as vocab
from contextlib import contextmanager
import shutil
from siganalogies import Dataset2016, Dataset2019
from typing import Union, Optional
import torch
from .data import pad
import logging
from pytorch_lightning.utilities import rank_zero_only

# Metric =======================================================================
# Cosine distance
def closest_cosine(query: torch.Tensor, voc: vocab.Vectors, ranks=False):
    if query.dim() == 1:
        similarities = torch.cosine_similarity(voc.vectors.to(query.device), query.unsqueeze(0), dim = -1)
        if ranks:
            idcs = similarities.argsort(descending=True, dim=-1)
            return voc.itos[idcs[0]], idcs
        else:
            idx = similarities.argmax(dim=-1)
            return voc.itos[idx], idx
    
    else:
        similarities = torch.cosine_similarity(voc.vectors.to(query.device).unsqueeze(0), query.unsqueeze(1), dim = -1)
        if ranks:
            idcs = similarities.argsort(descending=True, dim=-1)
            return [voc.itos[idcs_[0]] for idcs_ in idcs], idcs
        else:
            idx = similarities.argmax(dim=-1)
            return [voc.itos[idx_] for idx_ in idx], idx
        
# Euclidian distance
def closest_euclid(query: torch.Tensor, voc: vocab.Vectors, ranks=False):
    if query.dim() == 1:
        dists = torch.sqrt(((voc.vectors.to(query.device) - query) ** 2).sum(dim=-1))
    
        if ranks:
            idcs = dists.argsort(descending=False, dim=-1)
            return voc.itos[idcs[0]], idcs
        else:
            idx = dists.argmin(dim=-1)
            return voc.itos[idx], idx
    
    else:
        dists = torch.sqrt(((voc.vectors.to(query.device).unsqueeze(0) - query.unsqueeze(1)) ** 2).sum(dim=-1))

        if ranks:
            idcs = dists.argsort(descending=False, dim=-1)
            return [voc.itos[idcs_[0]] for idcs_ in idcs], idcs
        else:
            idx = dists.argmin(dim=-1)
            return [voc.itos[idx_] for idx_ in idx], idx


# 3CosAdd & 3CosMul ============================================================
def cosAdd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, voc: vocab.Vectors, ranks=False):
    return closest_cosine(b-a+c, voc, ranks=ranks)
def cosMul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, voc: vocab.Vectors, ranks=False, epsilon=1e-5):
    if a.dim() == 1:
        d = voc.vectors.to(a.device)
        da = torch.cosine_similarity(d, a.unsqueeze(0), dim = -1)
        db = torch.cosine_similarity(d, b.unsqueeze(0), dim = -1)
        dc = torch.cosine_similarity(d, c.unsqueeze(0), dim = -1)

        similarities=(db*dc)/(da+epsilon)

        if ranks:
            idcs = similarities.argsort(descending=True, dim=-1)
            return voc.itos[idcs[0]], idcs
        else:
            idx = similarities.argmin(dim=-1)
            return voc.itos[idx], idx
    
    else:
        d = voc.vectors.to(a.device).unsqueeze(0)
        da = torch.cosine_similarity(d, a.unsqueeze(1), dim = -1)
        db = torch.cosine_similarity(d, b.unsqueeze(1), dim = -1)
        dc = torch.cosine_similarity(d, c.unsqueeze(1), dim = -1)
        similarities = (db*dc)/(da+epsilon)

        if ranks:
            idcs = similarities.argsort(descending=True, dim=-1)
            return [voc.itos[idcs_[0]] for idcs_ in idcs], idcs
        else:
            idx = similarities.argmax(dim=-1)
            return [voc.itos[idx_] for idx_ in idx], idx

# Wrapper
def closest(query, voc: vocab.Vectors, ranks=False, strategy="cosine"):
    if strategy == "cosine":
        return closest_cosine(query, voc, ranks)
    elif strategy == "3CosMul":
        return cosMul(*query, voc, ranks)
    elif strategy == "3CosAdd":
        return cosAdd(*query, voc, ranks)
    else:
        return closest_euclid(query, voc, ranks)

def precision_sak_ap_rr(prediction: torch.Tensor, target: torch.Tensor, voc: vocab.Vectors, k=10, strategy="cosine"):
    """Computes: precision, success@k, rank, reciprocal rank, closest word to prediction, closest word to target"""
    if (isinstance(prediction, torch.Tensor) and prediction.dim() > 1) or (prediction[0].dim() > 1): # batch mode
        return b_precision_sak_ap_rr(prediction, target, voc, k=k, strategy=strategy)

    pred_w, idcs_pred = closest(prediction, voc, ranks=True, strategy=strategy)
    tgt_w, idx_tgt = closest(target, voc, strategy="cosine" if strategy in {"3CosAdd", "3CosMul"} else strategy)
    precision = 1. if idcs_pred[0] == idx_tgt else 0.
    if isinstance(k, list) or isinstance(k, tuple):
        sak = [
            1. if idcs_pred[:k_] in idx_tgt else 0.
            for k_ in k
        ]
    else:
        sak = 1. if idcs_pred[:k] in idx_tgt else 0.
    r = (idx_tgt == idcs_pred).nonzero(as_tuple=True)[0].item() + 1
    rr = 1/r

    return precision, sak, r, rr, pred_w, tgt_w

# Batch Metric =================================================================
def b_precision_sak_ap_rr(predictions: torch.Tensor, targets: torch.Tensor, voc: vocab.Vectors, k=10, strategy="cosine"):
    """Computes: precision, success@k, rank, reciprocal rank, closest word to prediction, closest word to target"""
    pred_w, idcs_pred = closest(predictions, voc, ranks=True, strategy=strategy)
    tgt_w, idx_tgt = closest(targets, voc, strategy="cosine" if strategy in {"3CosAdd", "3CosMul"} else strategy)
    precision = (idx_tgt == idcs_pred[:,0]).float()#.mean()

    if isinstance(k, list) or isinstance(k, tuple):
        sak = [
            (idcs_pred[:,:k_] == idx_tgt.unsqueeze(-1)).any(dim=-1).float()
            for k_ in k
        ]
    else:
        sak = (idcs_pred[:,:k] == idx_tgt.unsqueeze(-1)).any(dim=-1).float()#.mean()
    r = (idx_tgt.unsqueeze(-1) == idcs_pred).nonzero(as_tuple=True)[1] + 1
    rr = 1/r

    return precision, sak, r, rr, pred_w, tgt_w


# Embedding vocabulary generation ==============================================

@contextmanager
def embeddings_voc(embedding_model, train_dataset: Union[Dataset2016,Dataset2019], test_dataset: Optional[Union[Dataset2016,Dataset2019]]=None, *args, distributed_barier=True,**kwds):
    """Context manager building and returning the embeddings but then cleaning up the files in /tmp.
    
    Warning: If a word contains non-breaking spaces, they will be interpreted as regular spaces when loading."""
    try:
        check_dir(RETRIEVAL_CACHE)
        vectors_path = RETRIEVAL_VECTORS_PATH.format(language=train_dataset.language, mode=train_dataset.mode)
        check_dir(os.path.dirname(vectors_path))

        @rank_zero_only
        def _():
            return generate_embeddings_file(embedding_model=embedding_model, train_dataset=train_dataset, test_dataset=test_dataset,
            vectors_path=vectors_path)
        embeddings = _()
        if distributed_barier: torch.distributed.barrier(group=torch.distributed.group.WORLD) # just to have a barrier, such that non rank 0 processes have access to the vectors file
        embeddings = embeddings or vocab.Vectors(name = vectors_path, cache = RETRIEVAL_CACHE, unk_init = torch.Tensor.normal_)
        # fix internal vocab (replace non-braking spaces back into actual spaces)
        embeddings.itos = [w.replace(NONBREAKING_SPACE, ' ') for w in embeddings.itos]
        embeddings.stoi = {word: i for i, word in enumerate(embeddings.itos)}
        yield embeddings
    finally:
        @rank_zero_only
        def _():
            if RETRIEVAL_TMP.startswith("/tmp/") and os.path.exists(RETRIEVAL_TMP):
                shutil.rmtree(RETRIEVAL_TMP)
                logging.getLogger(__name__).info(f"Sucessfully removed temp folder {RETRIEVAL_TMP}")
        _()

#NONBREAKING_SPACE = b'\xc2\xa0'
NONBREAKING_SPACE = u"\u00A0"

def generate_embeddings_file(embedding_model,
    train_dataset: Union[Dataset2016,Dataset2019],
    test_dataset: Optional[Union[Dataset2016,Dataset2019]]=None,
    vectors_path = RETRIEVAL_VECTORS_PATH,
    recompute_vectors=True,
    encoder=None) -> vocab.Vectors:
    """Stores the embeddings of the encoder vocabulary and returns the path to the file.
    
    Warning: If a word contains spaces, they are replaced by non-breaking spaces in the file.
        If a word contains non-breaking spaces, they will be interpreted as regular spaces when loading with `embeddings_voc`.
    """
    
    if encoder is None: encoder = train_dataset.word_encoder

    if recompute_vectors:
        word_voc = train_dataset.word_voc
        if test_dataset is not None:
            word_voc = word_voc.union(test_dataset.word_voc)

        vocabulary = {word: train_dataset.word_encoder.encode(word) for word in word_voc}

        with open(vectors_path, 'w') as f:
            for word, embed in vocabulary.items():
                embedding = torch.unsqueeze(embed, 0)
                embedding = embedding_model(pad(embedding, train_dataset.word_encoder.BOS_ID, train_dataset.word_encoder.EOS_ID))
                embedding = torch.squeeze(embedding)
                embedding = embedding.tolist()
                embedding = ' '.join(str(i) for i in embedding)
                f.write(f"{word.replace(' ', NONBREAKING_SPACE)} {embedding}\n") # replace spaces into non-breaking spaces, such that when loading there is no issue

    embeddings = vocab.Vectors(name = vectors_path, cache = RETRIEVAL_CACHE, unk_init = torch.Tensor.normal_)

    return embeddings