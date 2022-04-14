import random
import numpy
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from siganalogies import dataset_factory, SIG2016_LANGUAGES, SIG2019_HIGH
from functools import partial
import logging

logger = logging.getLogger(__name__)

SPLIT_RANDOM_SEED = 42

def pad(tensor, bos_id, eos_id, target_size=-1):
    '''Adds a padding symbol at the beginning and at the end of a tensor.

    Arguments:
    tensor -- The tensor to pad.
    bos_id -- The value of the padding for the beginning.
    eos_id -- The value of the padding for the end.'''
    tensor = F.pad(input=tensor, pad=(1,0), mode='constant', value=bos_id)
    tensor = F.pad(input=tensor, pad=(0,1), mode='constant', value=eos_id)
    if target_size > 0 and tensor.size(-1) < target_size:
        tensor = F.pad(input=tensor, pad=(0,target_size - tensor.size(-1)), mode='constant', value=-1)

    return tensor

def collate(batch, bos_id, eos_id):
    '''Generates padded tensors of quadruples for the dataloader.

    Arguments:
    batch -- The original data.
    bos_id -- The value of the padding for the beginning.
    eos_id -- The value of the padding for the end.'''

    a_emb, b_emb, c_emb, d_emb = [], [], [], []

    len_a = max(len(a) for a, b, c, d in batch)
    len_b = max(len(b) for a, b, c, d in batch)
    len_c = max(len(c) for a, b, c, d in batch)
    len_d = max(len(d) for a, b, c, d in batch)

    for a, b, c, d in batch:
        a_emb.append(pad(a, bos_id, eos_id, len_a+2))
        b_emb.append(pad(b, bos_id, eos_id, len_b+2))
        c_emb.append(pad(c, bos_id, eos_id, len_c+2))
        d_emb.append(pad(d, bos_id, eos_id, len_d+2))

    # make a tensor of all As, af all Bs, of all Cs and of all Ds
    a_emb = torch.stack(a_emb)
    b_emb = torch.stack(b_emb)
    c_emb = torch.stack(c_emb)
    d_emb = torch.stack(d_emb)

    return a_emb, b_emb, c_emb, d_emb

def collate_words(batch, bos_id, eos_id):
    '''Generates padded tensors for the dataloader.

    Arguments:
    batch -- The original data.
    bos_id -- The value of the padding for the beginning.
    eos_id -- The value of the padding for the end.'''

    a_emb = []

    len_a = max(len(a) for a in batch)

    # make a tensor of all As, af all Bs, of all Cs and of all Ds
    a_emb = torch.stack([pad(a, bos_id, eos_id, len_a+2) for a in batch])

    return a_emb

def prepare_dataset(dataset_year, language, nb_analogies_train, nb_analogies_val, nb_analogies_test, force_rebuild=False, split_seed=SPLIT_RANDOM_SEED, force_low=False):
    '''Prepare the data for a given language.

    Arguments:
    language -- The language of the data to use for the training.
    nb_analogies_train -- The number of analogies to use (before augmentation) for the training.
    nb_analogies_val -- The number of analogies to use (before augmentation) for the validation (during the training).
    nb_analogies_test -- The number of analogies to use (before augmentation) for the testing (after the training).'''

    ## Train and test dataset
    if not force_low and ((dataset_year == "2019" and language in SIG2019_HIGH) or (dataset_year == "2016" and language=="japanese")):

        mode = "train" if dataset_year == "2016" else "train-high"
        dataset = dataset_factory(dataset=dataset_year, language=language, mode=mode, word_encoder="char", force_rebuild=force_rebuild)
        
        lengths = [nb_analogies_train, nb_analogies_val, nb_analogies_test]
        if sum(lengths) > len(dataset):
             # 75% for training (70% for the actual training data, 5% for devlopment) and 25% for testing
            lengths = [int(len(dataset) * .70), int(len(dataset) * .05)]
            lengths.append(len(dataset) - sum(lengths)) # add the remaining data for testing
            lengths.append(0) # add a chunk with the remaining unused data
            logger.warning(f"{language.capitalize()} is too small for the split {nb_analogies_train}|{nb_analogies_val}|{nb_analogies_test}, using {lengths[0]} (70%)|{lengths[1]} (5%)|{lengths[2]} (25%) instead.")
        else:
            lengths.append(len(dataset) - sum(lengths)) # add a chunk with the remaining unused data

        train_data, val_data, test_data, unused_data = random_split(dataset, lengths,
            generator=torch.Generator().manual_seed(split_seed))

    else:
        dataset = dataset_factory(dataset=dataset_year, language=language, mode="train" if dataset_year == "2016" else "train-low", word_encoder="char", force_rebuild=force_rebuild)
        dataset_dev = dataset_factory(dataset=dataset_year, language=language, mode="dev", word_encoder="char", force_rebuild=force_rebuild)
        dataset_test = dataset_factory(dataset=dataset_year, language=language, mode="test", word_encoder="char", force_rebuild=force_rebuild)

        train_data, unused_data = random_split(dataset, [nb_analogies_train, len(dataset) - nb_analogies_train],
            generator=torch.Generator().manual_seed(split_seed))
        val_data, unused_data = random_split(dataset_dev, [nb_analogies_val, len(dataset_dev) - nb_analogies_val],
            generator=torch.Generator().manual_seed(split_seed))
        test_data, unused_data = random_split(dataset_test, [nb_analogies_test, len(dataset_test) - nb_analogies_test],
            generator=torch.Generator().manual_seed(split_seed))

    return train_data, val_data, test_data, dataset

def prepare_data(dataset_year, language, nb_analogies_train, nb_analogies_val, nb_analogies_test, batch_size = 32, force_rebuild=False, generator_seed=42, split_seed=SPLIT_RANDOM_SEED, force_low=False):
    '''Prepare the dataloaders for a given language.

    Arguments:
    language -- The language of the data to use for the training.
    nb_analogies_train -- The number of analogies to use (before augmentation) for the training.
    nb_analogies_val -- The number of analogies to use (before augmentation) for the validation (during the training).
    nb_analogies_test -- The number of analogies to use (before augmentation) for the testing (after the training).'''
    train_data, val_data, test_data, dataset = prepare_dataset(dataset_year, language, nb_analogies_train, nb_analogies_val, nb_analogies_test, force_rebuild=force_rebuild, split_seed=split_seed, force_low=force_low)
    # Load data
    args = {
        "collate_fn": partial(collate, bos_id = dataset.word_encoder.BOS_ID, eos_id = dataset.word_encoder.EOS_ID),
        "num_workers": 4,
        "batch_size": batch_size,
        "persistent_workers": True
    }

    train_loader = DataLoader(train_data, generator=torch.Generator().manual_seed(generator_seed), shuffle=True, **args)
    val_loader = DataLoader(val_data, **args)#, generator=g_val)
    test_loader = DataLoader(test_data, **args)#, generator=g_test)

    return train_loader, val_loader, test_loader, dataset.word_encoder

