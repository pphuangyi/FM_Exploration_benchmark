import math
import torch
from torch import nn

class HingeLoss(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, head, tail, labels, return_dist):
        num_features = head.size(1)

        dist = torch.linalg.norm(head - tail, dim=-1) / math.sqrt(num_features)
        dist = torch.pow(dist, self.p)

        coef = 2. * labels - 1.

        loss = (coef * dist).mean()

        if return_dist:
            return loss, dist

        return loss
