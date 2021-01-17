import torch.nn as nn


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):

    def forward(self, x):
        return x.reshape(x.shape[0], -1)