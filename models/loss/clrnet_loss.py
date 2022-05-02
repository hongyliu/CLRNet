import torch
from torch import Tensor, tensor
import torch.nn as nn
from .focal_loss import SoftmaxFocalLoss
from .liou_loss import LIoU_Loss
import torch.nn.functional as F


class CLRNet_Loss(nn.Module):
    def __init__(self, num_points, gamma, radius, w_cls, w_sim, l_w_cls, l_w_xytl, l_w_liou):
        super(CLRNet_Loss, self).__init__()
        self.gamma = gamma
        self.radius = radius
        self.num_points = num_points
        self.w_cls = w_cls
        self.w_sim = w_sim
        self.l_w_cls = l_w_cls
        self.l_w_xytl = l_w_xytl
        self.l_w_liou = l_w_liou
        self.focal_loss = SoftmaxFocalLoss(gamma)
        self.liou_loss = LIoU_Loss(radius)
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, output: Tensor, label: Tensor):
        label_targets, \
        predict_features, \
        indices = self.get_assignments(output, label)

        sl1l_list = []
        ll_list = []
        fl_list = []
        for idx in range(len(label_targets)):
            dim0, dim1, dim2 = predict_features[idx].shape
            normal = F.normalize(torch.cat([predict_features[idx][:, :, 2:6], label_targets[idx][..., 2:6]], dim=1))
            fl_idx = []
            sl1l_idx = []
            for i in range(dim1):
                fl_idx.append(self.focal_loss(predict_features[idx][:, i, :2], torch.argmax(label_targets[idx][..., :2].squeeze(), keepdim=True)))
                sl1l_idx.append(self.smooth_l1_loss(normal[:, i, :], normal[:, -1, :]))
            fl_list.append(tensor(fl_idx, requires_grad=True).mean())

            ll_list.append(self.liou_loss(predict_features[idx][..., 6:], label_targets[idx][..., 6:]).mean())
            sl1l_list.append(tensor(sl1l_idx, requires_grad=True).mean())
        sl1l = tensor(sl1l_list, requires_grad=True).mean()
        ll = tensor(ll_list, requires_grad=True).mean()
        fl = tensor(fl_list, requires_grad=True).mean()
        # loss = self.l_w_cls * fl + self.l_w_xytl * sl1l + self.l_w_liou * ll
        losses = {}
        loss = 0.
        if sl1l > 0:
            losses['xytl_loss'] = self.l_w_xytl * sl1l
        if ll > 0:
            losses['liou_loss'] = self.l_w_liou * ll
        if fl > 0:
            losses['cls_loss'] = self.l_w_cls * fl

        for key, value in losses.items():
            loss += value
        losses['loss'] = loss
        losses['loss_states'] = losses
        return losses

    def get_assignments(self, output: Tensor, label: Tensor):
        label_targets = []
        predict_features = []
        indices_list = []
        for batch_idx in range(output.shape[0]):
            lanes_label = label[batch_idx:batch_idx + 1, :, :]
            predict = output[batch_idx:batch_idx + 1, :, :]
            for lane_idx in range(label.shape[1]):
                assign_cost = self.assign_cost(lanes_label[:, lane_idx:lane_idx + 1, :], predict)
                liou = self.liou_loss(predict[:, :, 6:], lanes_label[:, lane_idx:lane_idx + 1, 6:])
                topk = torch.clamp(torch.sum(liou).int(), min=1, max=64)
                _, indices = torch.topk(-assign_cost, topk.int())
                label_targets.append(lanes_label[:, lane_idx:lane_idx + 1, :])
                predict_features.append(predict[:, indices, :])
                indices_list.append(indices)
                # assign_liou_loss.append(liou[:, indices])
        return label_targets, predict_features, indices_list

    def assign_cost(self, lane_prior, feature):
        focal_cost = self.focal_cost(feature[..., :2], lane_prior[..., :2], self.gamma).reshape(-1)
        xy_cost = self.xy_cost(feature[..., 3:5], lane_prior[..., 3:5]).reshape(-1)
        theta_cost = self.theta_cost(feature[..., 5], lane_prior[..., 5]).reshape(-1)
        dis_cost = self.dis_cost(feature[..., 6:], lane_prior[..., 6:]).reshape(-1)
        return self.w_sim * torch.pow(dis_cost * xy_cost * theta_cost, 2) + self.w_cls * focal_cost

    def dis_cost(self, logits, labels):
        cost = torch.sum(torch.abs(logits - labels), 2) / self.num_points
        return F.normalize(cost).unsqueeze(0)

    def theta_cost(self, logits, labels):
        cost = (logits - labels).unsqueeze(2)
        return F.normalize(cost).unsqueeze(0)

    def xy_cost(self, logits, labels):
        cost = torch.sqrt(torch.sum(torch.pow(logits - labels, 2), dim=2))
        return F.normalize(cost).unsqueeze(0)

    def focal_cost(self, logits, labels, gamma):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        cost = torch.sum(log_score * labels, dim=2).squeeze()
        return cost
