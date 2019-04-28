import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForwardLayer
from utils import clone_layer


class Encoder(nn.Module):
    def __init__(self, device, stack_size=6, d_model=512):
        super(Encoder, self).__init__()
        self.device = device
        self.layers = clone_layer(EncoderLayer(d_model).to(device), stack_size)

    def forward(self, input, mask=None):
        for layer in self.layers:
            input = layer(input, mask)
        return input


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, p_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.sub_layer1 = MultiHeadAttention(p_dropout)
        self.sub_layer2 = FeedForwardLayer(p_dropout)

    def forward(self, input, mask=None):
        out = self.sub_layer1(input, input, input, mask)
        return self.sub_layer2(out)
