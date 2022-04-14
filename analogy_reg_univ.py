import torch
import torch.nn as nn


class AnalogyRegression(nn.Module):
    def __init__(self, filters=16, **kwargs):
        super().__init__()

        self.abac = nn.Sequential(nn.Conv2d(1, filters, (1,2), stride=(1,2)), nn.ReLU())
        self.d = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 0)),
            nn.Conv2d(filters, 1, (2,2), stride=(1,1))
        )

    def forward(self, a, b, c, p=0):
        """
        Expected input shape for each of a, b, and c: [batch_size, emb_size]
        Output shape: [batch_size, emb_size]
        """

        if p>0:
            a=torch.nn.functional.dropout(a, p)
            b=torch.nn.functional.dropout(b, p)
            c=torch.nn.functional.dropout(c, p)

        # [batch_size, emb_size] * 4 ->  [batch_size, 1, emb_size, 4]
        input =  torch.stack([a, b, a, c], dim = -1).unsqueeze(1)

        # [batch_size, 1, emb_size, 4] ->  [batch_size, filters, emb_size, 2]
        abac = self.abac(input)
        # [batch_size, filters, emb_size, 2] ->  [batch_size, 1, emb_size, 1]
        d = self.d(abac)

        # [batch_size, 1, emb_size, 1] -> [batch_size, emb_size]
        d = d.squeeze(3).squeeze(1)
        return d

if __name__ == "__main__":
    a,b,c,d = torch.randn((4,80)), torch.randn((4,80)), torch.randn((4,80)), torch.randn((4,80))
    reg = AnalogyRegression()

    print(reg(a,b,c).size())

