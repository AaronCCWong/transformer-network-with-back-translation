import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForwardLayer
from helpers import clone_layer


class Decoder(nn.Module):
    def __init__(self, stack_size):
        super(Decoder, self).__init__()
        self.layers = clone_layer(DecoderLayer, stack_size)

    def forward(self, input):
        for layer in layers:
            input = layer(input)
        return input


class DecoderLayer(nn.Module):
    def __init__(self, p_dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.sub_layer1 = MultiHeadAttention()
        self.sub_layer2 = MultiHeadAttention()
        self.sub_layer3 = FeedForwardLayer()
        self.layer_norm = nn.LayerNorm()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, input):
        residual = input
        out = self.dropout(self.sub_layer1(input))
        out = self.layer_norm(residual + out)

        residual = out
        out2 = self.dropout(self.sub_layer2(out))
        out2 = self.layer_norm(residual + out2)

        residual = out2
        out3 = self.dropout(self.sub_layer3(out2))
        return self.layer_norm(residual + out3)
