import torch.nn as nn


class FeedForwardLayer(nn.Module):
    def __init__(self, p_dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(512, 2048, 1),
            nn.ReLU(),
            nn.Conv1d(2048, 512, 1),
            nn.Dropout(p_dropout))
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, input):
        residual = input
        out = input.transpose(1, 2)
        out = self.net(out)
        out = out.transpose(1, 2)
        return self.layer_norm(out + residual)
