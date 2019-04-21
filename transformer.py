import torch.nn as nn
from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.encoder = Encoder(6)
        self.decoder = Decoder(6)
        self.linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax()

    def forward(self, src, tgt, src_mask, tgt_mask):
        out = self.src_embedding(src)
        encoded_input = self.encoder(out, src_mask)
        return self.decoder(src, encoded_input, src_mask, tgt, tgt_mask)
