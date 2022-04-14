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
from utils.ret import closest_cosine, vocab, closest_euclid, b_precision_sak_ap_rr, precision_sak_ap_rr

vectors_path = "testing/vect"

class TestRetrivalUtils(unittest.TestCase):
    def make_dummy(self):
        with open(vectors_path, 'w') as f:
            for idx, embed in enumerate(torch.randn((1000,32))):
                embedding = embed.tolist()
                embedding = ' '.join(str(i) for i in embedding)
                f.write(f"{idx} {embedding}\n")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.make_dummy()
        self.voc = vocab.Vectors("testing/vect")

    def test_cos(self):
        x=torch.randn((5,32))
        ans_0, idx_0 = closest_cosine(x[0], self.voc, True)
        ans_1, idx_1 = closest_cosine(x[0], self.voc, False)
        ans_2, idx_2 = closest_cosine(x, self.voc, True)
        ans_3, idx_3 = closest_cosine(x, self.voc, False)
        self.assertEqual(idx_0[0], idx_1)
        self.assertListEqual(idx_0.tolist(), idx_2[0].tolist())
        self.assertEqual(idx_1, idx_3[0])

        self.assertEqual(ans_0, ans_1)
        self.assertEqual(ans_0, ans_2[0])
        self.assertEqual(ans_0, ans_3[0])
        self.assertListEqual(ans_2, ans_3)

    def test_euclid(self):
        x=torch.randn((5,32))
        ans_0, idx_0 = closest_euclid(x[0], self.voc, True)
        ans_1, idx_1 = closest_euclid(x[0], self.voc, False)
        ans_2, idx_2 = closest_euclid(x, self.voc, True)
        ans_3, idx_3 = closest_euclid(x, self.voc, False)
        self.assertEqual(idx_0[0], idx_1)
        self.assertListEqual(idx_0.tolist(), idx_2[0].tolist())
        self.assertEqual(idx_1, idx_3[0])

        self.assertEqual(ans_0, ans_1)
        self.assertEqual(ans_0, ans_2[0])
        self.assertEqual(ans_0, ans_3[0])
        self.assertListEqual(ans_2, ans_3)
        self.assertIsInstance(idx_2, torch.Tensor)

    def test_b_precision_sak_ap_rr(self):
        x=torch.randn((5,32))
        y=torch.randn((5,32))
        b_precision, b_sak, b_r, b_rr, b_pred_w, b_tgt_w = b_precision_sak_ap_rr(x,y,self.voc)
        
        for i in range(5):
            precision, sak, r, rr, pred_w, tgt_w = precision_sak_ap_rr(x[i],y[i],self.voc)
            self.assertEqual(b_precision[i], precision)
            self.assertEqual(b_sak[i], sak)
            self.assertEqual(b_r[i], r)
            self.assertEqual(b_rr[i], rr)
            self.assertEqual(b_pred_w[i], pred_w)
            self.assertEqual(b_tgt_w[i], tgt_w)


if __name__ == "__main__":
    unittest.main()
