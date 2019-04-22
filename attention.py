import torch
import numpy as np
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, device):
        super(MultiHeadAttention, self).__init__()
        self.device = device

        self.v_linear = nn.Linear(512, 512).to(device)
        self.k_linear = nn.Linear(512, 512).to(device)
        self.q_linear = nn.Linear(512, 512).to(device)

        self.scaled_dot_product_att = ScaledDotProductAttention()
        self.linear = nn.Linear(512, 512)

    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.size(0)

        queries = self.q_linear(queries).view(batch_size, -1, 8, 64).transpose(1, 2)
        keys = self.k_linear(keys).view(batch_size, -1, 8, 64).transpose(1, 2)
        values = self.v_linear(values).view(batch_size, -1, 8, 64).transpose(1, 2)

        out = self.scaled_dot_product_att(queries, keys, values, mask)
        concat = out.transpose(1, 2).contiguous().view(batch_size, -1, 512)
        return self.linear(concat)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, queries, keys, values, mask=None):
        d_k = queries.size(-1)
        qk_t = torch.matmul(queries, keys.transpose(-2, -1))
        scaled_qk_t = qk_t / np.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scaled_qk_t = scaled_qk_t.masked_fill(mask==0, -1e9)

        attention = self.softmax(scaled_qk_t)
        return torch.matmul(attention, values)
