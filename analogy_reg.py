import torch
import torch.nn as nn


class AnalogyRegression(nn.Module):
    def __init__(self, emb_size, mode = "ab!=ac", **kwargs):
        """Model:
        - d = f3(f1(a, b), f2(a, c))
        
        :param mode: if equal to "ab=ac", f1 will be used in place of f2: 
            d = f3(f1(a, b), f1(a, c))
        """
        super().__init__()
        self.emb_size = emb_size
        self.ab = nn.Linear(2 * self.emb_size, 2 * self.emb_size)
        if mode == "ab=ac":
            self.ac = self.ab
        else:
            self.ac = nn.Linear(2 * self.emb_size, 2 * self.emb_size)
        self.d = nn.Linear(4 * self.emb_size, self.emb_size)

    def forward(self, a, b, c, p=0):

        if p>0:
            a=torch.nn.functional.dropout(a, p)
            b=torch.nn.functional.dropout(b, p)
            c=torch.nn.functional.dropout(c, p)

        ab = self.ab(torch.cat([a, b], dim = -1))
        ac = self.ac(torch.cat([a, c], dim = -1))

        d = self.d(torch.cat([ab, ac], dim = -1))
        return d
