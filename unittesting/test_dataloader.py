# Import the os module
import itertools
import os, sys
from random import randint
import unittest

import torch
from torch.utils.data import DataLoader

# Change the current working directory
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, enrich, generate_negative, n_pos_n_neg
from utils import prepare_data
from pickle import dumps

class TestGenerator(unittest.TestCase):
    def test_generator_pickle(self):
        dataloader=DataLoader(
            torch.randn((5,6,4), generator=torch.Generator().manual_seed(42))
        )
        dumps(dataloader)
                        
if __name__ == "__main__":
    unittest.main()
