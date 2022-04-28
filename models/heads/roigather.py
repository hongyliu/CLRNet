import math

import torch
from ..registry import HEADS
from torch import nn, Tensor


def generate_uniform_prior(batch, lanes, prior_elements, points, img_w, start=0):
    prior = torch.rand(batch, lanes, prior_elements)
    prior = (img_w - start) * prior + start
    prior[:, :, 0], prior[:, :, 1], prior[:, :, 2], prior[:, :, 5] = 1, 0, points, 0
    return prior


def prior_add_y(prior: Tensor, points, img_h):
    prior_y = Tensor([img_h - 1 - img_h / (points - 1) * i for i in range(points)]).resize(1, 1, points)
    prior_y = torch.cat((prior_y, prior_y))
    prior = torch.hstack((prior, prior_y))
    return prior


def lane_roi_align(feature: Tensor, lane_prior: Tensor, num_points, sample_points, feature_h, feature_w, img_h) -> Tensor:
    ratio = feature_h / img_h
    prior = lane_prior * ratio
    sample_stride = num_points // sample_points
    sample_prior = prior[:, :, ::sample_stride]
    batch, lanes = lane_prior.shape[0], lane_prior.shape[1] - 1
    feature_batch_list = list()
    for i in range(batch):
        feature_lane_list = list()
        for j in range(lanes):
            feature_select_list = list()
            for k in range(sample_points):
                x_sample, y_sample = sample_prior[i, j, k], sample_prior[i, lanes, k]
                x_sample = x_sample if x_sample <= feature_w - 1 else feature_w - 1
                y_sample = y_sample if y_sample <= feature_h - 1 else feature_h - 1
                feature_select = (feature[i, :, math.ceil(y_sample), math.ceil(x_sample)] +
                                 feature[i, :, math.ceil(y_sample), math.floor(x_sample)] +
                                 feature[i, :, math.floor(y_sample), math.ceil(x_sample)] +
                                 feature[i, :, math.floor(y_sample), math.floor(x_sample)]) / 4
                feature_select_list.append(feature_select.reshape(1, -1, 1, 1))
            feature_lane_list.append(torch.cat(feature_select_list, dim=3))
        feature_batch_list.append(torch.cat(feature_lane_list, dim=2))
    return torch.cat(feature_batch_list, dim=0).permute(0, 2, 1, 3)


class ROIGatherBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 points,
                 sample_points,
                 feature_hw,
                 resize_shape,
                 prior_elements,
                 max_lanes,
                 img_h,
                 batch_size):
        super(ROIGatherBlock, self).__init__()
        self.in_channel = in_channel
        self.points = points
        self.sample_points = sample_points
        self.resize_shape = resize_shape
        self.resize_h = resize_shape[0]
        self.resize_w = resize_shape[1]
        self.feature_h = feature_hw[0]
        self.feature_w = feature_hw[1]
        self.prior_elements = prior_elements
        self.max_lanes = max_lanes
        self.img_h = img_h
        self.batch_size = batch_size
        self.resize_flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((10, 25)),
            nn.Flatten(start_dim=-2, end_dim=-1))
        self.conv_fc = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=5),
            nn.Flatten(),
            nn.Linear(in_channel * 32, in_channel)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channel, self.prior_elements))
        self.conv_fc.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def forward(self, feature_input, prior_input):
        prior = prior_add_y(prior_input[:, :, 6:], self.points, self.img_h)
        roi = lane_roi_align(feature_input, prior, self.points, self.sample_points, self.feature_h,
                             self.feature_w, self.img_h)
        x_f = torch.unsqueeze(self.resize_flatten(feature_input), 1).repeat(1, self.max_lanes, 1, 1)
        x_p = self.conv_fc(roi.reshape(-1, self.in_channel, self.sample_points))\
            .reshape(-1, self.max_lanes, 1, self.in_channel)
        w = self.softmax(torch.matmul(x_p, x_f) / math.sqrt(self.in_channel))
        g = torch.matmul(w, x_f.permute(0, 1, 3, 2))
        p = g.reshape(self.batch_size, -1, 1) + prior_input
        # p = self.fc((g + x_p).reshape(-1, self.in_channel)).reshape(-1, self.max_lanes, self.prior_elements)
        return p

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.5)


@HEADS.register_module
class ROIGather(nn.Module):
    def __init__(self,
                 in_channels,
                 feature_size,
                 feature_channel,
                 sample_points,
                 resize_shape,
                 cfg=None):
        super(ROIGather, self).__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.feature_channel = feature_channel
        self.sample_points = sample_points
        self.resize_shape = resize_shape
        self.cfg = cfg
        self.roi_layers = nn.ModuleList()
        for i, in_channel in enumerate(in_channels):
            roi_gather_block = ROIGatherBlock(in_channel,
                                              self.cfg.num_points,
                                              self.sample_points,
                                              self.feature_size[i],
                                              self.resize_shape,
                                              self.cfg.prior_elements,
                                              self.cfg.max_lanes,
                                              self.cfg.img_h,
                                              self.cfg.batch_size)
            self.roi_layers.append(roi_gather_block)

    def forward(self, feature_list, batch):
        generate_prior = generate_uniform_prior(self.cfg.batch_size, self.cfg.max_lanes, self.cfg.prior_elements,
                                                self.cfg.num_points, self.cfg.img_w)
        prior_output = []
        for i, roi_gather_block in enumerate(self.roi_layers):
            prior_input = generate_prior if i == 0 else prior_output[-1]
            prior_refined = roi_gather_block(feature_list[i], prior_input)
            prior_output.append(prior_refined)
        return prior_output



# if __name__ == '__main__':
#     prior = generate_uniform_prior(2, 5, 78, 72, 800)
#     # prior = prior_add_y(prior[:, :, 6:], 72, 320)
#     feature = torch.randn(2, 64, 40, 100)
#     # roi = lane_roi_align(feature, prior, 72, 36, 40, 100, 320)
#
#     block = ROIGatherBlock(64, 72, 36, (40, 100), (10, 25), 78, 5, 320)
#     data = block(feature, prior)
#     print(data)