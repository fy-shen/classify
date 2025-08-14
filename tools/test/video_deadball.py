import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed, Logger, Video
from utils.build import Builder
from utils.file import splitfn, load_label_map


blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


def best_by_roc(y_true, y_score, save_path):
    from sklearn.metrics import roc_curve, accuracy_score, auc
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_roc = auc(fpr, tpr)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]

    y_pred = np.asarray(y_score >= best_thresh, int)
    acc = accuracy_score(y_true, y_pred)
    if save_path:
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve')
        plt.scatter(fpr[best_idx], tpr[best_idx], c='red', label=f'Best Threshold={best_thresh:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(save_path)
        plt.close()
    return auc_roc, best_thresh, acc


def best_by_pr(y_true, y_score, save_path):
    from sklearn.metrics import precision_recall_curve, accuracy_score, auc
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    auc_pr = auc(recalls, precisions)

    f1_scores = 2 * precisions[1:] * recalls[1:] / (precisions[1:] + recalls[1:] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    y_pred = np.asarray(y_score >= best_thresh, int)
    acc = accuracy_score(y_true, y_pred)
    if save_path:
        plt.figure()
        plt.plot(recalls, precisions, label='PR curve')
        plt.scatter(recalls[best_idx + 1], precisions[best_idx + 1], c='red', label=f'Best Threshold={best_thresh:.3f}')
        plt.xlabel("R")
        plt.ylabel("P")
        plt.title("PR Curve")
        plt.legend()
        plt.savefig(save_path)
        plt.close()
    return auc_pr, best_thresh, acc


def video_infer(gpu_id, video_file, label_file, model, trans, label_map, logger, cfg):
    save_array_path = os.path.join(cfg.save_dir, 'arrays', f'{splitfn(video_file)[1]}.npy')
    if os.path.exists(save_array_path):
        return
    target_names = [label_map[i] for i in range(len(label_map))]
    video = Video(video_file)
    writer = None
    if cfg.test.save_vid:
        save_vid_path = os.path.join(cfg.save_dir, os.path.basename(video_file))
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_vid_path, fourcc, video.fps, (video.w, video.h))

    if os.path.exists(label_file):
        labels = np.load(label_file)['label']
        use_label = True
        video_len = min(video.frames, len(labels))
    else:
        labels = None
        use_label = False
        video_len = video.frames

    frames_raw, labels_raw, inputs = [], [], []
    conf_all, pred_all, label_all = [], [], []
    pbar = tqdm(range(video_len), desc=f'[DeadBall Demo {os.path.split(video_file)[-1]}]', ncols=100)
    with torch.no_grad():
        for i in pbar:
            ret, frame_raw = video.cap.read()
            if not ret:
                break
            frames_raw.append(frame_raw)
            if use_label:
                labels_raw.append(labels[i])
            if len(frames_raw) % cfg.test.frame_stride == 0:
                image_choice = frames_raw[len(frames_raw) - cfg.test.frame_stride // 2]
                image = Image.fromarray(cv2.cvtColor(image_choice, cv2.COLOR_BGR2RGB))
                inputs.append(image)
            if len(inputs) == model.num_seg:
                # [T,C,H,W]
                data = trans(inputs)
                data = data.unsqueeze(0).to(gpu_id, non_blocking=True)
                output = model(data)
                confs = F.softmax(output, dim=1)[0]
                pred = torch.argmax(confs).item()
                for idx in range(len(frames_raw)):
                    conf_all.append(confs.cpu().numpy())
                    pred_all.append(pred)
                    if use_label:
                        label_all.append(labels_raw[idx])
                    if cfg.test.save_vid:
                        frame = frames_raw[idx]
                        for clsid, conf in enumerate(confs):
                            text = f'{label_map[clsid]}: {conf:.2f}'
                            # 推理预测对应类别红色，其余类别蓝色
                            color = red if pred == clsid else blue
                            cv2.putText(frame, text, (video.w//2, 50 * (clsid + 1)), cv2.FONT_HERSHEY_TRIPLEX,
                                        2, color, 1, cv2.LINE_AA, bottomLeftOrigin=False)
                            if use_label:
                                # 标签类别前画一个绿点
                                if labels_raw[idx] == clsid:
                                    cv2.circle(frame, (video.w//2 - 25, 30 + clsid * 50), 10, green,
                                               thickness=-1, lineType=cv2.LINE_AA)
                        writer.write(frame)

                frames_raw.clear(), labels_raw.clear(), inputs.clear()

        # save confs
        np.save(save_array_path,
                np.asarray(conf_all),
                allow_pickle=True)

        if cfg.test.save_vid:
            writer.release()

        if use_label:
            from sklearn.metrics import (
                confusion_matrix, ConfusionMatrixDisplay,
                classification_report, precision_recall_fscore_support,
            )

            filtered_labels = []
            filtered_confs = []
            filtered_preds = []
            for gt, conf, pred in zip(label_all, conf_all, pred_all):
                if gt != -1:
                    filtered_labels.append(gt)
                    filtered_confs.append(conf)
                    filtered_preds.append(pred)

            # 0.5阈值分类报告
            report = classification_report(filtered_labels, filtered_preds, target_names=target_names, digits=3)
            logger.log(logger.make_separator(os.path.split(video_file)[-1]))
            logger.log(report)

            # 混淆矩阵
            precision, recall, f1, support = precision_recall_fscore_support(
                filtered_labels, filtered_preds, labels=list(range(len(target_names)))
            )
            logger.log(f"{'Class':<15}{'TP':<8}{'FP':<8}{'P':<8}{'R':<8}{'F1':<8}{'Support'}")
            cm = confusion_matrix(filtered_labels, filtered_preds, labels=list(range(len(target_names))))
            for i, name in enumerate(target_names):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                logger.log(f"{name:<15}{tp:<8}{fp:<8}{precision[i]:<8.3f}{recall[i]:<8.3f}{f1[i]:<8.3f}{support[i]}")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
            disp.plot(include_values=True, cmap='Blues', xticks_rotation='horizontal')
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(cfg.save_dir, f'{splitfn(video_file)[1]}_confusion_matrix.png'))
            plt.close()

            # ROC, PR
            y_true = np.asarray(filtered_labels)
            y_score = np.asarray(filtered_confs)[:, 1]
            roc_path = os.path.join(cfg.save_dir, f'{splitfn(video_file)[1]}_roc.png')
            pr_path = os.path.join(cfg.save_dir, f'{splitfn(video_file)[1]}_pr.png')
            auc_roc, thresh_roc, acc_roc = best_by_roc(y_true, y_score, roc_path)
            auc_pr, thresh_pr, acc_pr = best_by_pr(y_true, y_score, pr_path)
            logger.log(f"\n{'Curve':<10}{'AUC':<12}{'BestThr':<12}{'ACC@Thr':<12}")
            logger.log(f"{'ROC':<10}{auc_roc:<12.3f}{thresh_roc:<12.3f}{acc_roc:<12.3f}")
            logger.log(f"{'PR':<10}{auc_pr:<12.3f}{thresh_pr:<12.3f}{acc_pr:<12.3f}")


def main(cfg):
    os.makedirs(os.path.join(cfg.save_dir, 'arrays'), exist_ok=True)

    logger = Logger(cfg)
    logger.log_cfg(cfg)
    set_random_seed(cfg.SEED, cfg.DETERMINISTIC)
    gpu_id = cfg.GPU_IDS[0]

    builder = Builder(cfg, logger)
    model = builder.build_model('test').to(gpu_id)
    model.eval()
    if cfg.GPU_NUM > 1:
        model = nn.DataParallel(model, device_ids=cfg.GPU_IDS)

    trans = builder.build_transform(is_train=False)
    label_map = load_label_map(os.path.join(cfg.data_params.root_path, cfg.data_params.class_map))

    video_path = os.path.join(cfg.data_params.root_path, cfg.test.video_path)
    label_path = os.path.join(cfg.data_params.root_path, cfg.test.label_path)
    if os.path.isdir(video_path):
        for video_f in sorted(os.listdir(video_path)):
            video_file = os.path.join(video_path, video_f)
            fn = splitfn(video_file)[1]
            label_file = os.path.join(label_path, fn + '.npz')
            video_infer(gpu_id, video_file, label_file, model, trans, label_map, logger, cfg)
    elif os.path.isfile(video_path):
        video_file = video_path
        fn = splitfn(video_file)[1]
        label_file = os.path.join(label_path, fn + '.npz') if os.path.isdir(label_path) else label_path
        video_infer(gpu_id, video_file, label_file, model, trans, label_map, logger, cfg)
    else:
        raise ValueError(f"{video_path} is not a valid file or directory.")


