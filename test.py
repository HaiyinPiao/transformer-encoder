import math
import time
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
from transformer.Models import Transformer, Encoder
from transformer.Optim import ScheduledOptim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_stacks = Encoder(d_model=32, d_inner=64,
            n_layers=2, n_head=4, d_k=16, d_v=16, dropout=0.1)

criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.SGD(encoder_stacks.parameters(), lr=1)

src = torch.rand(1, 2, 32,requires_grad=True)
tgt = torch.rand(1, 2, 32)

print(src)

encoder_stacks.train()

for i in range(100):
    out, attn = encoder_stacks.forward(src, src_mask = None)    
    loss = criterion(out, tgt)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder_stacks.parameters(), 0.5)
    optimizer.step()
    print(loss.item())

print("out:", out)
print("tgt:", tgt)
print("attn:", attn)