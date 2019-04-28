import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForwardLayer
from utils import clone_layer


class Decoder(nn.Module):
    def __init__(self, device, stack_size=6, d_model=512):
        super(Decoder, self).__init__()
        self.layers = clone_layer(DecoderLayer(d_model).to(device), stack_size)

    def forward(self, input, encoded_input, src_mask, tgt_mask):
        for layer in self.layers:
            input = layer(input, encoded_input, src_mask, tgt_mask)
        return input


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, p_dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.sub_layer1 = MultiHeadAttention(p_dropout)
        self.sub_layer2 = MultiHeadAttention(p_dropout)
        self.sub_layer3 = FeedForwardLayer(p_dropout)

    def forward(self, input, encoded_input, src_mask, tgt_mask):
        out = self.sub_layer1(input, input, input, tgt_mask)
        out2 = self.sub_layer2(out, encoded_input, encoded_input, src_mask)
        return self.sub_layer3(out2)
