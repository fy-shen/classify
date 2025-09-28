import os
from omegaconf import OmegaConf
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from archs import register


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


@register('dataset')
class TSNDataset(data.Dataset):
    def __init__(self, cfg, is_train, transform):
        model_cfg = OmegaConf.load(cfg.model_cfg)
        self.cfg = OmegaConf.merge(cfg, model_cfg)
        self.is_train = is_train
        self.transform = transform
        self.num_seg = self.cfg.num_seg
        self.modality = self.cfg.modality

        data_params = self.cfg.data_params
        self.root_path = data_params.root_path
        self.list_file = data_params.train_list if is_train else data_params.val_list
        self.image_tmpl = data_params.image_tmpl
        self.dense_sample = data_params.dense_sample
        self.clip_len = data_params.clip_len
        self.num_sample = data_params.num_sample
        self.new_length = data_params.new_length
        self.twice_sample = data_params.twice_sample

        self._parse_list()

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(os.path.join(self.root_path, self.list_file))]
        tmp = [item for item in tmp if int(item[1]) >= 3]  # filter out short clips
        self.video_list = [VideoRecord(item) for item in tmp]

    def _get_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - self.clip_len)
            t_stride = self.clip_len // self.num_seg
            if self.num_sample > 1:
                start_list = np.linspace(0, sample_pos - 1, num=self.num_sample, dtype=int)
                offsets = []
                for start_idx in start_list:
                    offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_seg)]
            else:
                start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
                offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_seg)]

        else:
            avg_duration = (record.num_frames - self.new_length + 1) // self.num_seg
            if self.is_train:
                if avg_duration > 0:
                    offsets = np.multiply(list(range(self.num_seg)), avg_duration) + \
                              np.random.randint(avg_duration, size=self.num_seg)
                else:
                    offsets = np.linspace(0, max(record.num_frames - self.new_length, 0), self.num_seg, dtype=int)
            else:
                if record.num_frames > self.num_seg + self.new_length - 1:
                    if self.twice_sample:
                        offsets = np.concatenate([
                            np.array([int(avg_duration / 2.0 + avg_duration * x) for x in range(self.num_seg)]),
                            np.array([int(avg_duration * x) for x in range(self.num_seg)])
                        ])
                    else:
                        offsets = np.array([int(avg_duration / 2.0 + avg_duration * x) for x in range(self.num_seg)])
                else:
                    offsets = np.zeros((self.num_seg,))
        return offsets + 1

    def _load_image(self, directory, idx):
        img_path = os.path.join(self.root_path, directory, self.image_tmpl.format(idx))
        img_backup = os.path.join(self.root_path, directory, self.image_tmpl.format(1))
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(img_path).convert('RGB')]
            except Exception:
                print('Error loading image:', img_path)
                return [Image.open(img_backup).convert('RGB')]
        elif self.modality == 'Flow':
            # TODO:
            pass

    def __getitem__(self, index):
        record = self.video_list[index]
        seg_indices = self._get_indices(record)
        return self.get(record, seg_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        # [T,C,H,W]
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


class FrameRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def idx(self):
        return int(self._data[1])

    @property
    def label(self):
        return [float(self._data[2]), float(self._data[3])]


@register('dataset')
class TPDataset(data.Dataset):
    def __init__(self, cfg, is_train, transform):
        model_cfg = OmegaConf.load(cfg.model_cfg)
        self.cfg = OmegaConf.merge(cfg, model_cfg)
        self.is_train = is_train
        self.transform = transform
        self.num_seg = self.cfg.num_seg

        data_params = self.cfg.data_params
        self.root_path = data_params.root_path
        self.list_file = data_params.train_list if is_train else data_params.val_list
        self.image_tmpl = data_params.image_tmpl
        self.gap = data_params.gap

        self._parse_list()

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(os.path.join(self.root_path, self.list_file))]
        self.frame_list = [FrameRecord(item) for item in tmp]

    def _get_indices(self, record):
        offsets = []
        max_idx = len(os.listdir(os.path.join(self.root_path, 'images', record.path))) - 1
        for i in range(0 - self.num_seg // 2, self.num_seg - self.num_seg // 2):
            idx = record.idx + i * self.gap
            if idx < 0:
                idx = 0
            if idx > max_idx:
                idx = max_idx
            offsets.append(idx + 1)
        return offsets

    def _load_image(self, directory, idx):
        img_path = os.path.join(self.root_path, 'images', directory, self.image_tmpl.format(idx))
        img_backup = os.path.join(self.root_path, 'images', directory, self.image_tmpl.format(1))
        try:
            return [Image.open(img_path).convert('RGB')]
        except Exception:
            print('Error loading image:', img_path)
            return [Image.open(img_backup).convert('RGB')]

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index):
        record = self.frame_list[index]
        indices = self._get_indices(record)
        return self.get(record, indices)

    def trans_label(self, label):
        x, y = label
        rh, rw = self.cfg.input_size
        ph, pw = self.cfg.pad_size
        x = (x * rw + (pw - rw) // 2) / pw
        y = (y * rh + (ph - rh) // 2) / ph
        return [x, y]

    def get(self, record, indices):
        images = list()
        for idx in indices:
            seg_imgs = self._load_image(record.path, idx)
            images.extend(seg_imgs)

        # [T,C,H,W]
        process_data = self.transform(images)
        return process_data, torch.tensor(self.trans_label(record.label))



