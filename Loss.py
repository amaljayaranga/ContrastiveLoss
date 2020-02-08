import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9


    def forward(self, output1, output2, target):
        eq_distance = F.pairwise_distance(output1, output2)
        loss = 0.5 * (1 - target.float()) * torch.pow(eq_distance, 2) + \
               0.5 * target.float() * torch.pow(torch.clamp(self.margin - eq_distance, min=0.00), 2)
        return loss.mean()

