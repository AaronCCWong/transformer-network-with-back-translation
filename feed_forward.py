import torch.nn as nn


class FeedForwardLayer(nn.Module):
    def __init__(self):
        super(FeedForwardLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512))

    def forward(self, input):
        return self.net(input)
