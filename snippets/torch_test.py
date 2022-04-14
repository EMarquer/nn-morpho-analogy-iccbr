import torch

x, y = torch.randn((1,2,5)), torch.randn((1,2,5))
x.cuda()
y.cuda()

print((x*y).mean())