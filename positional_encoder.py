import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoder(nn.Module):
    def __init__(self, device, d_model=512, max_seq_len=50):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout()

        self.positional_encoder = torch.zeros(max_seq_len, d_model).to(device)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.positional_encoder[pos, i] = math.sin(self.frequencies(pos, i, d_model))
                self.positional_encoder[pos, i+1] = math.cos(self.frequencies(pos, i+1, d_model))

    def forward(self, embedding_input):
        sequence_len = embedding_input.size(1)
        positional_encoding = self.positional_encoder[:, :sequence_len]
        embedding_input = embedding_input + Variable(positional_encoding, requires_grad=False)
        return self.dropout(embedding_input)

    def frequencies(self,  pos, i, d_model=512):
        return pos / (10000 ** (2 * i / d_model))
