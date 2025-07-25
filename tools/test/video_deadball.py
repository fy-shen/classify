import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed, Logger, Video
from utils.build import Builder
from utils.file import splitfn, load_label_map


blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


def video_infer(gpu_id, video_file, label_file, model, trans, label_map, cfg):
    video = Video(video_file)
    result_path = os.path.join(cfg.save_dir, os.path.basename(video_file))
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(result_path, fourcc, video.fps, (video.w, video.h))

    if os.path.exists(label_file):
        label = np.load(label_file)['label']
        use_label = True
        video_len = min(video.frames, len(label))
    else:
        label = None
        use_label = False
        video_len = video.frames

    frames_raw, labels_raw, inputs = [], [], []
    pred_all, label_all = [], []
    pbar = tqdm(range(video_len), desc=f'[DeadBall Demo {os.path.split(video_file)[-1]}]', ncols=100)
    with torch.no_grad():
        for i in pbar:
            ret, frame_raw = video.cap.read()
            if not ret:
                break
            frames_raw.append(frame_raw)
            if use_label:
                labels_raw.append(label[i])
            if i % cfg.test.frame_stride == 0:
                image = Image.fromarray(cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB))
                inputs.append(image)
            if len(inputs) == model.num_seg:
                # [T,C,H,W]
                data = trans(inputs)
                data = data.unsqueeze(0).to(gpu_id)
                output = model(data)
                confs = F.softmax(output, dim=1)[0]
                pred = torch.argmax(confs).item()
                for idx in range(len(frames_raw)):
                    frame = frames_raw[idx]
                    for clsid, conf in enumerate(confs):
                        text = f'{label_map[clsid]}: {conf:.2f}'
                        # 推理预测对应类别红色，其余类别蓝色
                        color = red if pred == clsid else blue
                        cv2.putText(frame, text, (1920, 50 * (clsid + 1)), cv2.FONT_HERSHEY_TRIPLEX,
                                    2, color, 1, cv2.LINE_AA, bottomLeftOrigin=False)
                        if use_label:
                            # 标签类别前画一个绿点
                            if labels_raw[idx] == clsid:
                                cv2.circle(frame, (1920 - 25, 30 + clsid * 50), 10, green,
                                           thickness=-1, lineType=cv2.LINE_AA)
                    writer.write(frame)
                    if use_label:
                        pred_all.append(pred)
                        label_all.append(labels_raw[idx])
                frames_raw.clear(), labels_raw.clear(), inputs.clear()

        writer.release()
        if use_label:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib.pyplot as plt

            cm = confusion_matrix(label_all, pred_all)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_map)
            disp.plot(include_values=True, cmap='Blues', xticks_rotation='vertical')
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(cfg.save_dir, f'{splitfn(video_file)[1]}_confusion_matrix.png'))
            plt.close()


def main(cfg):
    logger = Logger(cfg)
    logger.log_cfg(cfg)
    set_random_seed(cfg.SEED, cfg.DETERMINISTIC)
    gpu_id = cfg.GPU_IDS[0]

    builder = Builder(cfg, logger)
    model = builder.build_model('test').to(gpu_id)
    if cfg.GPU_NUM > 1:
        model = nn.DataParallel(model, device_ids=cfg.GPU_IDS)

    trans = builder.build_transform(is_train=False)
    label_map = load_label_map(os.path.join(cfg.data_params.root_path, cfg.data_params.class_map))

    video_path = os.path.join(cfg.data_params.root_path, cfg.test.video_path)
    label_path = os.path.join(cfg.data_params.root_path, cfg.test.label_path)
    if os.path.isdir(video_path):
        for video_f in os.listdir(video_path):
            video_file = os.path.join(video_path, video_f)
            fn = splitfn(video_file)[1]
            label_file = os.path.join(label_path, fn + '.npz')
            video_infer(gpu_id, video_file, label_file, model, trans, label_map, cfg)
    elif os.path.isfile(video_path):
        video_file = video_path
        fn = splitfn(video_file)[1]
        label_file = os.path.join(label_path, fn + '.npz') if os.path.isdir(label_path) else label_path
        video_infer(gpu_id, video_file, label_file, model, trans, label_map, cfg)
    else:
        raise ValueError(f"{video_path} is not a valid file or directory.")


