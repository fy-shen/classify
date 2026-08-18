import os
import random
from dataclasses import dataclass
from omegaconf import OmegaConf
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from archs import register


def _parse_time_to_frame(text, fps):
    parts = text.strip().split(':')
    if len(parts) == 3:
        h, m, s = parts
        seconds = int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        seconds = int(m) * 60 + float(s)
    else:
        seconds = float(parts[0])
    return int(round(seconds * fps)) + 1


@dataclass
class ShotEvent:
    video: str
    frame: int
    half: str


@dataclass
class ShotSample:
    video: str
    center: int
    half: str
    label: int


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


@register('dataset')
class ShotTSMDataset(data.Dataset):
    def __init__(self, cfg, is_train, transform):
        model_cfg = OmegaConf.load(cfg.model_cfg)
        self.cfg = OmegaConf.merge(cfg, model_cfg)
        self.is_train = is_train
        self.transform = transform
        self.num_seg = self.cfg.num_seg
        self.modality = self.cfg.modality

        data_params = self.cfg.data_params
        self.root_path = data_params.root_path
        self.images_dir = os.path.join(self.root_path, data_params.get('images_dir', 'images'))
        self.ann_dir = os.path.join(self.root_path, data_params.get('annotation_dir', 'annotations/shot'))
        self.image_tmpl = data_params.image_tmpl
        self.fps = data_params.get('fps', 30)
        self.frame_stride = data_params.get('frame_stride', 2)
        self.pos_jitter = data_params.get('pos_jitter', self.frame_stride)
        self.pos_radius = data_params.get('pos_radius', self.num_seg * self.frame_stride // 2)
        self.neg_exclude_radius = max(
            data_params.get('neg_exclude_radius', self.num_seg * self.frame_stride),
            self.pos_radius
        )
        self.neg_per_pos = data_params.get('neg_per_pos', 1)
        self.val_neg_per_pos = data_params.get('val_neg_per_pos', self.neg_per_pos)
        self.val_neg_stride = data_params.get('val_neg_stride', self.num_seg * self.frame_stride)
        self.min_frames = data_params.get('min_frames', self.num_seg * self.frame_stride + 1)
        self.train_list = data_params.get('train_list', None)
        self.val_list = data_params.get('val_list', None)
        self.val_ratio = data_params.get('val_ratio', 0.2)
        self.split_seed = data_params.get('split_seed', 2025)

        self.events_by_video, self.frame_counts = self._load_events()
        self.pos_samples = self._build_positive_samples()
        if not self.pos_samples:
            raise ValueError(f"No valid shot annotations found in {self.ann_dir}.")
        self.neg_samples = self._build_negative_samples(self.val_neg_per_pos if not is_train else self.neg_per_pos)
        self.samples = self.pos_samples + self.neg_samples
        self.cls_names = ['background', 'shot']

    def _load_events(self):
        events_by_video = {}
        frame_counts = {}
        for fn in sorted(os.listdir(self.ann_dir)):
            if not fn.endswith('.txt'):
                continue
            video = os.path.splitext(fn)[0]
            image_dir = os.path.join(self.images_dir, video)
            if not os.path.isdir(image_dir):
                continue
            frame_count = len([x for x in os.listdir(image_dir) if x.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if frame_count < self.min_frames:
                continue
            frame_counts[video] = frame_count
            events = []
            with open(os.path.join(self.ann_dir, fn), 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    half = parts[1].upper()
                    if half not in {'L', 'R'}:
                        continue
                    frame = min(max(_parse_time_to_frame(parts[0], self.fps), 1), frame_count)
                    events.append(ShotEvent(video, frame, half))
            if events:
                events_by_video[video] = events
        videos = self._select_videos(sorted(events_by_video))
        events_by_video = {v: events_by_video[v] for v in videos}
        frame_counts = {v: frame_counts[v] for v in videos}
        return events_by_video, frame_counts

    def _read_video_list(self, list_file):
        path = os.path.join(self.root_path, list_file)
        videos = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = line.strip().split()
                if item:
                    videos.append(os.path.splitext(os.path.basename(item[0]))[0])
        return videos

    def _select_videos(self, videos):
        list_file = self.train_list if self.is_train else self.val_list
        if list_file:
            selected = self._read_video_list(list_file)
            return [v for v in selected if v in videos]

        rng = random.Random(self.split_seed)
        videos = list(videos)
        rng.shuffle(videos)
        val_num = max(1, int(round(len(videos) * self.val_ratio))) if len(videos) > 1 else 0
        val_set = set(videos[:val_num])
        return [v for v in videos if (v not in val_set if self.is_train else v in val_set)]

    def _clip_center(self, center, frame_count):
        radius = (self.num_seg - 1) * self.frame_stride // 2
        return min(max(center, 1 + radius), max(1 + radius, frame_count - radius))

    def _build_positive_samples(self):
        samples = []
        for video, events in self.events_by_video.items():
            frame_count = self.frame_counts[video]
            for event in events:
                center = self._clip_center(event.frame, frame_count)
                samples.append(ShotSample(video, center, event.half, 1))
        return samples

    def _is_negative_center(self, video, center, half):
        for event in self.events_by_video.get(video, []):
            if event.half == half and abs(center - event.frame) <= self.neg_exclude_radius:
                return False
        return True

    def _build_negative_samples(self, neg_per_pos):
        samples = []
        rng = random.Random(2025 if not self.is_train else None)
        for video, events in self.events_by_video.items():
            frame_count = self.frame_counts[video]
            target = max(1, len(events) * neg_per_pos)
            candidates = []
            for half in ('L', 'R'):
                for center in range(1, frame_count + 1, self.val_neg_stride):
                    center = self._clip_center(center, frame_count)
                    if self._is_negative_center(video, center, half):
                        candidates.append(ShotSample(video, center, half, 0))
            if self.is_train:
                valid = candidates
                if not valid:
                    continue
                for _ in range(target):
                    samples.append(rng.choice(valid))
            else:
                samples.extend(candidates[:target])
        return samples

    def _sample_train_negative(self, sample):
        frame_count = self.frame_counts[sample.video]
        for _ in range(100):
            center = random.randint(1, frame_count)
            half = random.choice(('L', 'R'))
            if self._is_negative_center(sample.video, center, half):
                return ShotSample(sample.video, self._clip_center(center, frame_count), half, 0)
        return sample

    def _get_indices(self, center, frame_count):
        if self.is_train:
            center += random.randint(-self.pos_jitter, self.pos_jitter)
        center = self._clip_center(center, frame_count)
        offsets = np.arange(self.num_seg) - self.num_seg // 2
        indices = center + offsets * self.frame_stride
        return np.clip(indices, 1, frame_count).astype(int)

    def _load_image(self, directory, idx):
        img_path = os.path.join(self.images_dir, directory, self.image_tmpl.format(idx))
        img_backup = os.path.join(self.images_dir, directory, self.image_tmpl.format(1))
        try:
            return [Image.open(img_path).convert('RGB')]
        except Exception:
            print('Error loading image:', img_path)
            return [Image.open(img_backup).convert('RGB')]

    @staticmethod
    def _crop_half(images, half):
        half = half.upper()
        if half not in {'L', 'R'}:
            return images
        result = []
        for img in images:
            w, h = img.size
            box = (0, 0, w // 2, h) if half == 'L' else (w // 2, 0, w, h)
            result.append(img.crop(box))
        return result

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.is_train and sample.label == 0:
            sample = self._sample_train_negative(sample)

        images = []
        for idx in self._get_indices(sample.center, self.frame_counts[sample.video]):
            images.extend(self._load_image(sample.video, idx))

        images = self._crop_half(images, sample.half)
        process_data = self.transform(images)
        return process_data, sample.label

    def __len__(self):
        return len(self.samples)


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
