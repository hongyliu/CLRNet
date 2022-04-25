import os.path as osp
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import logging
from .registry import DATASETS
from .process import Process
from utils.visualization import imshow_lanes


@DATASETS.register_module
class BaseDataset(Dataset):
    def __init__(self, data_root, split, processes=None,
            cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = 'train' in split 
        self.processes = Process(processes, cfg)


    def view(self, predictions, img_meta):
        img_names = img_meta['img_name']
        for lanes, img_name in zip(predictions, img_names):
            img = cv2.imread(osp.join(self.data_root, img_name))
            out_file = osp.join(self.cfg.work_dir, 'visualization', img_name.replace('/', '_'))
            lanes = [lane.to_array(self.cfg) for lane in lanes]
            imshow_lanes(img, lanes, out_file=out_file)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info['img_path'])#.astype(np.float32)
        img = img[self.cfg.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cfg.cut_height:, :]
            sample.update({'mask': label})

        sample = self.processes(sample)
        # meta = {'full_img_path': data_info['img_path'],
        #         'img_name': data_info['img_name']}
        # sample.update({'meta': meta})

        return sample 
