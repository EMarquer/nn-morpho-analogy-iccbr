from multiprocessing import Pool
from torch.utils.data import random_split
from siganalogies import dataset_factory
from siganalogies.config import SIG2016_LANGUAGES
from lepage_sigmorphon import LepageSigmorphonDataset
from collections import Counter
from statistics import mean, stdev
import pandas as pd
import sys

dataset_lepage = None
dataset_train = None
dataset_test = None
def in_lepage(x):
    a,b,c,d = x
    return 1 if (
    (a,b,c,d) in dataset_lepage or
    (c,d,a,b) in dataset_lepage
    ) else 0
def in_train(x):
    a,b,c,d = x
    return 1 if any((a_,b_,c_,d_) in dataset_train for a_,b_,c_,d_ in permute(a,b,c,d)) else 0
def in_test(x):
    a,b,c,d = x
    return 1 if any((a_,b_,c_,d_) in dataset_test for a_,b_,c_,d_ in permute(a,b,c,d)) else 0

def permute(a,b,c,d):
    yield a, b, c, d
    yield c, d, a, b
    yield c, a, d, b
    yield d, b, c, a
    yield d, c, b, a
    yield b, a, d, c
    yield b, d, a, c
    yield a, c, b, d

for lang in SIG2016_LANGUAGES:
    if lang != "japanese":
        print(f"Starting {lang}...")
        print("Loading Sigmorphon...")
        dataset_train = dataset_factory(lang, "train")
        dataset_train = frozenset(dataset_train)
        dataset_test = dataset_factory(lang, "test")
        dataset_test = frozenset(dataset_test)
        print("Loading Lepage...")
        dataset_lepage = LepageSigmorphonDataset(lang)
        dataset_lepage = frozenset(dataset_lepage.data)
        print(next(iter(dataset_test)))
        print(next(iter(dataset_lepage)))

        print("Starting processing...")
        p = Pool()
        base_results_test = p.map(in_lepage, dataset_test)
        coverage_test = mean(base_results_test)
        p.close()
        p = Pool()
        base_results_train = p.map(in_lepage, dataset_train)
        coverage_train = mean(base_results_train)
        p.close()
        p = Pool()
        base_results_train_inverse = p.map(in_train, dataset_lepage)
        coverage_train_inverse = mean(base_results_train_inverse)
        p.close()
        p = Pool()
        base_results_test_inverse = p.map(in_test, dataset_lepage)
        coverage_test_inverse = mean(base_results_test_inverse)
        p.close()
        print(f"{lang}: Lepage covers {coverage_train:.2%} of train and {coverage_test:.2%} of test")
        print(f"{lang}: Train covers {coverage_train_inverse:.2%} of Lepage")
        print(f"{lang}: Test covers {coverage_test_inverse:.2%} of Lepage")

