from misc import *
import os
import numpy as np
from tqdm import tqdm
import cv2

pred_dir = 'xxxx'
gt_dir = 'xxxx'

print(pred_dir)
print('doing,please wait for a moment')

num_files = len(os.listdir(pred_dir))
progress_bar = tqdm(total=num_files)

def resize_masks(pred_mask, gt_mask):
    pred_mask_resized = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    # gt_mask_resized = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred_mask_resized, gt_mask

# Initialize evaluation metric lists outside the loop
acc_l = []
iou_l = []
mae_l = []
ber_l = []
precision_record, recall_record = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]

for name in os.listdir(os.path.join(pred_dir)):
    gt = get_gt_mask(name, gt_dir)
    normalized_pred = get_normalized_predict_mask(name, pred_dir)
    binary_pred = get_binary_predict_mask(name, pred_dir)

    if normalized_pred.ndim == 3:
        normalized_pred = normalized_pred[:, :, 0]
    if binary_pred.ndim == 3:
        binary_pred = binary_pred[:, :, 0]

    # Ensure binary_pred and normalized_pred have the same shape
    if binary_pred.shape != normalized_pred.shape:
        binary_pred, normalized_pred = resize_masks(binary_pred, normalized_pred)

    # PMD 数据集
    if len(gt.shape) == 3: # 检查gt是否有3个维度（x, y, z）
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) # 将gt转换为灰度图像（x, y）
    else:
        gt = gt # 如果gt已经是(x, y)格式，则不进行转换
        
    # Resize masks to have the same shape as gt
    binary_pred, gt = resize_masks(binary_pred, gt)
    normalized_pred, gt = resize_masks(normalized_pred, gt)
    # print(binary_pred.shape,normalized_pred.shape,gt.shape)
    # Perform evaluation

    # acc_l.append(compute_acc_mirror(binary_pred, gt))
    iou_l.append(compute_iou(binary_pred, gt))
    mae_l.append(compute_mae(normalized_pred, gt))
    # ber_l.append(compute_ber(binary_pred, gt))

    pred = (255 * normalized_pred).astype(np.uint8)
    gt = (255 * gt).astype(np.uint8)
    p, r = cal_precision_recall(pred, gt)
    for idx, data in enumerate(zip(p, r)):
        p, r = data
        precision_record[idx].update(p)
        recall_record[idx].update(r)

    progress_bar.update(1)

progress_bar.close()

print('%s:  mae: %4f, ber: %4f, acc: %4f, iou: %4f, f_measure: %4f' %
        (pred_dir, np.mean(mae_l), np.mean(ber_l), np.mean(acc_l), np.mean(iou_l),
        cal_fmeasure([precord.avg for precord in precision_record], [rrecord.avg for rrecord in recall_record])))