import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForwardLayer
from layer_norm import LayerNorm


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = []


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.sub_layer1 = MultiHeadAttention()
        self.sub_layer2 = FeedForwardLayer()
        self.layer_norm = LayerNorm()

    def forward(self, input):
        residual = input
        out = self.sub_layer1(input)
        out = self.layer_norm(residual + out)

        residual = out
        output = self.sub_layer2(out)
        return self.layer_norm(residual + output)
