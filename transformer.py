import torch.nn as nn
from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.encoder = Encoder(6)
        self.decoder = Decoder(6)
        self.linear = nn.Linear(512, 512)
        self.softmax = nn.Softmax()

    def forward(self):
        raise Exception("Not Implemented")
