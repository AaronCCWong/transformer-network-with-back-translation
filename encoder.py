import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForwardLayer
from helpers import clone_layer


class Encoder(nn.Module):
    def __init__(self, stack_size):
        super(Encoder, self).__init__()
        self.layers = clone_layer(EncoderLayer(), stack_size)

    def forward(self, input, mask=None):
        for layer in layers:
            input = layer(input, mask)
        return input


class EncoderLayer(nn.Module):
    def __init__(self, p_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.sub_layer1 = MultiHeadAttention()
        self.sub_layer2 = FeedForwardLayer()
        self.layer_norm = nn.LayerNorm()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, input, mask=None):
        residual = input
        out = self.dropout(self.sub_layer1(input, input, input, mask))
        out = self.layer_norm(residual + out)

        residual = out
        output = self.dropout(self.sub_layer2(out))
        return self.layer_norm(residual + output)
