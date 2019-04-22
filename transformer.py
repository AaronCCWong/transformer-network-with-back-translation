import math
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder
from positional_encoder import PositionalEncoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512):
        super(Transformer, self).__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(d_model)
        self.encoder = Encoder(6, d_model)
        self.decoder = Decoder(6, d_model)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, src, tgt, src_mask, tgt_mask):
        out = self.src_embedding(src) * math.sqrt(self.d_model)
        out = self.positional_encoder(out)
        encoded_input = self.encoder(out, src_mask)
        out = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        out = self.positional_encoder(out)
        out = self.decoder(out, encoded_input, src_mask, tgt_mask)
        out = self.linear(out)
        return self.softmax(out)
