# Import the os module
import itertools
import os, sys
from random import randint
import unittest

import torch

# Change the current working directory
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH, enrich, generate_negative, n_pos_n_neg
from utils import prepare_data


class TestDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_l = {'2016': SIG2016_LANGUAGES, '2019': SIG2019_HIGH}
        #self.d_l = {'2016': ["finnish"]}

    def test_reproductibe_loading(self):
        for dataset, langs in self.d_l.items():
            with self.subTest(dataset=dataset):
                for lang in langs:
                    for i in range(3):
                        seed = randint(0,999999999999)
                        with self.subTest(lang=lang, i=i):
                            train_loader, val_loader, test_loader, encoder = prepare_data(
                                dataset, lang, 50000, 500, 50000,
                                2, True, seed)
                            _train_loader, _val_loader, _test_loader, _encoder = prepare_data(
                                dataset, lang, 50000, 500, 50000,
                                2, False, seed)

                            self.assertTupleEqual(encoder.id_to_char, _encoder.id_to_char)

                            for subset, _subset, subset_name in ((train_loader, _train_loader, "train"), (val_loader, _val_loader, "val"), (test_loader, _test_loader, "test")):
                                with self.subTest(subset_name=subset_name):
                                    self.assertListEqual(train_loader.dataset.indices, _train_loader.dataset.indices)
                                    samples, _samples = list(itertools.islice(iter(train_loader), 5)), list(itertools.islice(iter(_train_loader), 5))
                                    
                                    for j in range(5):
                                        self.assertTrue((samples[j][0] == _samples[j][0]).all())
                                        self.assertTrue((samples[j][1] == _samples[j][1]).all())
                                        self.assertTrue((samples[j][2] == _samples[j][2]).all())
                                        self.assertTrue((samples[j][3] == _samples[j][3]).all())
                        
if __name__ == "__main__":
    unittest.main()
