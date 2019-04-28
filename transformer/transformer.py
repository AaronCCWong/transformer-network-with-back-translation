import math
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .positional_encoder import PositionalEncoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, device, d_model=512, p_dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoder1 = PositionalEncoder(device, d_model=d_model, p_dropout=p_dropout)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoder2 = PositionalEncoder(device, d_model=d_model, p_dropout=p_dropout)
        self.encoder = Encoder(device, 6, d_model)
        self.decoder = Decoder(device, 6, d_model)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        # Share weights
        self.linear.weight = self.tgt_embedding.weight

    def forward(self, src, tgt, src_mask, tgt_mask):
        out = self.src_embedding(src) * math.sqrt(self.d_model)
        out = self.positional_encoder1(out)
        encoded_input = self.encoder(out, src_mask)
        out = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        out = self.positional_encoder2(out)
        out = self.decoder(out, encoded_input, src_mask, tgt_mask)
        out = self.linear(out)
        return self.softmax(out)
