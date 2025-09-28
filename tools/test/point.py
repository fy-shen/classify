import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed, Logger, Video
from utils.build import Builder
from utils.file import splitfn


blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


def clamp(i: int, n_frames: int) -> int:
    return max(0, min(n_frames - 1, i))


def trans_coord(coord, video):
    coord = coord.cpu().numpy()
    coord = coord * [video.w, video.h]
    return coord[0]


def load_labels(label_file, video):
    if os.path.exists(label_file):
        data = np.load(label_file)
        labels = {}
        for obj in data:
            frame, x, y, _ = obj
            labels[int(frame)] = [x * video.w, y * video.h]
        return labels, True
    else:
        return {}, False


def draw_point(img, x, y, color):
    x, y = int(x), int(y)
    xl, xr = x - 100, x + 100
    yt, yb = y - 100, y + 100
    cv2.circle(img, (x, y), 5, color, -1)
    cv2.rectangle(img, (xl, yt), (xr, yb), color, 2)
    cv2.line(img, (xl, yt), (xr, yb), color, 2)
    cv2.line(img, (xl, yb), (xr, yt), color, 2)
    return img


def compute_error(preds, labels, logger):
    preds_, labels_ = [], []
    for f in preds.keys():
        if f in labels:
            preds_.append(preds[f])
            labels_.append(labels[f])

    if len(preds_) > 0:
        preds_ = np.stack(preds_)
        labels_ = np.stack(labels_)
        dists = np.linalg.norm(preds_ - labels_, axis=1)
        logger.log(f"{'MeanDist':<10}{dists.mean():.3f}")
        logger.log(f"{'Threshold':<12}{'Accuracy':<12}{'Num'}")
        for th in [5, 15, 30, 60, 120]:
            acc = (dists <= th).mean()
            correct = (dists <= th).sum()
            logger.log(f"<= {th:<4}px   {acc:<12.3%}{correct}")


def save_preds(preds, path, video):
    result = []
    for k in preds.keys():
        x, y = preds[k]
        x, y = x / video.w, y / video.h
        result.append([k, x, y, 1])
    np.save(path, np.array(result, dtype=np.float32))


def video_infer(gpu_id, video_file, label_file, model, trans, logger, cfg):
    save_array_path = os.path.join(cfg.save_dir, 'arrays', f'{splitfn(video_file)[1]}.npy')
    if os.path.exists(save_array_path):
        return
    video = Video(video_file)
    writer = None
    if cfg.test.save_vid:
        save_vid_path = os.path.join(cfg.save_dir, os.path.basename(video_file))
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_vid_path, fourcc, video.fps, (video.w, video.h))

    labels, use_label = load_labels(label_file, video)

    key_indices = list(range(0, video.frames, cfg.data_params.gap))  # 关键帧索引
    key_p = 0  # 关键帧索引指针
    offsets = (-14, -7, 0, 7, 14)
    frame_buffer = {}       # 帧缓存
    preds = {}              # 模型预测
    per_frame_preds = {}    # 逐帧预测
    prev_t, prev_coord, prev_head = None, None, 0
    infer_time, infer_num = 0, 0
    pbar = tqdm(range(video.frames), desc=f'[Point Demo {os.path.split(video_file)[-1]}]', ncols=100)
    with torch.no_grad():
        for idx in pbar:
            ret, frame = video.cap.read()
            frame_buffer[idx] = frame
            if not ret:
                break
            if key_p < len(key_indices):
                t = key_indices[key_p]
                idxs = [clamp(t + off, video.frames) for off in offsets]
                if all(i in frame_buffer for i in idxs):
                    frames_list = [Image.fromarray(cv2.cvtColor(frame_buffer[i], cv2.COLOR_BGR2RGB)) for i in idxs]
                    inputs = trans(frames_list).unsqueeze(0).to(gpu_id, non_blocking=True)
                    start = time.time()
                    coord, _ = model(inputs)
                    infer_time += time.time() - start
                    infer_num += 1
                    coord = trans_coord(coord, video)
                    preds[t] = coord

                    # 插值、绘图
                    if prev_t is not None:
                        x0, y0 = prev_coord
                        x1, y1 = coord
                        span = t - prev_t
                        if cfg.test.save_vid:
                            for f in range(prev_t, t + 1):
                                alpha = (f - prev_t) / span
                                x = x0 * (1 - alpha) + x1 * alpha
                                y = y0 * (1 - alpha) + y1 * alpha
                                per_frame_preds[f] = (x, y)

                                out_frame = frame_buffer[f]
                                xi, yi = int(round(x)), int(round(y))
                                out_frame = draw_point(out_frame, xi, yi, red)
                                if use_label and f in labels:
                                    lx, ly = labels[f]
                                    out_frame = draw_point(out_frame, lx, ly, blue)
                                writer.write(out_frame)

                        # 释放缓存
                        for f in range(prev_head, idxs[0]):
                            frame_buffer.pop(f, None)

                    prev_t, prev_coord, prev_head = t, coord, idxs[0]
                    key_p += 1

    save_preds(preds, save_array_path, video)
    if cfg.test.save_vid:
        writer.release()

    if use_label:
        logger.log(logger.make_separator(os.path.split(video_file)[-1]))
        compute_error(preds, labels, logger)

    logger.log(f"Average Inference Time: {infer_time / infer_num:.3f} sec")


def main(cfg):
    logger = Logger(cfg)
    logger.log_cfg(cfg)
    set_random_seed(cfg.SEED, cfg.DETERMINISTIC)
    gpu_id = cfg.GPU_IDS[0]

    builder = Builder(cfg, logger)
    model = builder.build_model('test').to(gpu_id)
    model.eval()

    trans = builder.build_transform(is_train=False)
    video_path = os.path.join(cfg.data_params.root_path, cfg.test.video_path)
    label_path = os.path.join(cfg.data_params.root_path, cfg.test.label_path)
    os.makedirs(os.path.join(cfg.save_dir, 'arrays'), exist_ok=True)
    if os.path.isdir(video_path):
        for video_f in sorted(os.listdir(video_path)):
            video_file = os.path.join(video_path, video_f)
            fn = splitfn(video_file)[1]
            label_file = os.path.join(label_path, fn + '.npy')
            video_infer(gpu_id, video_file, label_file, model, trans, logger, cfg)
    elif os.path.isfile(video_path):
        video_file = video_path
        fn = splitfn(video_file)[1]
        label_file = os.path.join(label_path, fn + '.npy') if os.path.isdir(label_path) else label_path
        video_infer(gpu_id, video_file, label_file, model, trans, logger, cfg)
    else:
        raise ValueError(f"{video_path} is not a valid file or directory.")
