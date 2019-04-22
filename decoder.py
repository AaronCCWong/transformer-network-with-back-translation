import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForwardLayer
from utils import clone_layer


class Decoder(nn.Module):
    def __init__(self, device, stack_size=6, d_model=512):
        super(Decoder, self).__init__()
        self.layers = clone_layer(DecoderLayer(device, d_model), stack_size)

    def forward(self, input, encoded_input, src_mask, tgt_mask):
        for layer in self.layers:
            input = layer(input, encoded_input, src_mask, tgt_mask)
        return input


class DecoderLayer(nn.Module):
    def __init__(self, device, d_model=512, p_dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.sub_layer1 = MultiHeadAttention(device)
        self.sub_layer2 = MultiHeadAttention(device)
        self.sub_layer3 = FeedForwardLayer()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, input, encoded_input, src_mask, tgt_mask):
        residual = input
        out = self.dropout(self.sub_layer1(input, input, input, tgt_mask))
        out = self.layer_norm(residual + out)

        residual = out
        out2 = self.dropout(self.sub_layer2(out, encoded_input, encoded_input, src_mask))
        out2 = self.layer_norm(residual + out2)

        residual = out2
        out3 = self.dropout(self.sub_layer3(out2))
        return self.layer_norm(residual + out3)
