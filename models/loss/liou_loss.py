import torch
from torch import Tensor
import torch.nn as nn
from models.loss.focal_loss import SoftmaxFocalLoss


class LIoU_Loss(nn.Module):
    def __init__(self, radius):
        super(LIoU_Loss, self).__init__()
        self.radius = radius

    def forward(self, predict:Tensor, ground_truth: Tensor):
        d_o = torch.abs(torch.min(predict + self.radius, ground_truth + self.radius) - \
                        torch.max(predict - self.radius, ground_truth - self.radius))
        d_u = torch.abs(torch.max(predict + self.radius, ground_truth + self.radius) - \
                        torch.min(predict - self.radius, ground_truth - self.radius))
        return 1 - torch.sum(d_o, dim=2) / torch.sum(d_u, dim=2)
