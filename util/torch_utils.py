import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedHuberLoss(nn.Module):
    ''' Compute weighted Huber loss for use with Pioritized Expereince Replay '''
    def __init__(self):
        super(WeightedHuberLoss, self).__init__()

    def forward(self, input, target, weights, mask):
        batch_size = input.size(0)
        batch_loss = (torch.abs(input - target) < 1).float() * (input - target)**2 + \
                     (torch.abs(input - target) >= 1).float() * (torch.abs(input - target) - 0.5)
        batch_loss *= mask
        weighted_batch_loss = weights * batch_loss.view(batch_size, -1).sum(dim=1)
        weighted_loss = weighted_batch_loss.sum() / batch_size

        return weighted_loss
