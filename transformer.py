import torch.nn as nn
from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(6)
        self.decoder = Decoder(6)
        self.linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax()

    def forward(self):
        raise Exception("Not Implemented")
