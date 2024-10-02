import torch
import numpy as np
from PIL import Image
import os

def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((200, 200))  # 确保图片大小为200x200
    image = np.array(image)
    return image

def convert_to_labels(image):
    # 将图片转换为标签图
    labels = np.zeros(image.shape[:2], dtype=np.int32)
    for class_id, color in enumerate([(0, 0, 1), (0, 0, 2), (0, 0, 3)]):
        mask = np.all(image == np.array(color)[:, None, None], axis=-1)
        labels[mask] = class_id + 1  # 类别从1开始

    return labels

def calculate_iou(pred, target, n_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred == target).long().sum().data.cpu().numpy()
    union = (pred != 0).long().sum().data.cpu().numpy() + (target != 0).long().sum().data.cpu().numpy() - intersection
    iou = intersection / union
    iou[np.isnan(iou)] = 0
    return np.mean(iou)

def calculate_miou(pred, target, n_classes):
    iou = []
    pred = pred.long()
    target = target.long()
    for cls in range(1, n_classes + 1):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).float().sum((1, 2))
        union = (pred_inds | target_inds).float().sum((1, 2))
        iou_cls = intersection / union
        iou.append((iou_cls.mean().item() if not np.isnan(iou_cls.mean().item()) else 0))
    return np.mean(iou)

# 假设第一张图片是类别0，第二张图片是类别1
# 这里我们使用简单的二分类示例，你需要根据实际情况调整
image1_path = '/home/SegContest/model/Pytorch-UNet/predict'  # 第一张图片的路径
image2_path = '/home/SegContest/model/NEU_Seg-main/annotations/test'  # 第二张图片的路径
pred = []
test = []
if __name__=="__main__":
    for filename in os.listdir(image1_path):
        if os.path.isfile(os.path.join(image1_path, filename)):
            pred.append(os.path.join(image1_path, filename))
            # out_files.append(os.path.join(args.output,filename))
    for filename in os.listdir(image2_path):
        if os.path.isfile(os.path.join(image2_path, filename)):
            test.append(os.path.join(image2_path, filename))

    for i in range(len(pred)):

        image1 = load_image(pred[i])
        image2 = load_image(test[i])

        # 将图片转换为二进制掩码（0和1）
        # 这里我们假设第一张图片是类别0，第二张图片是类别1
        # 你需要根据实际情况调整阈值和类别
        pred_labels = convert_to_labels(image1)
        target_labels = convert_to_labels(image2)

        # 计算MIoU
        n_classes = 3  # 类别数
        miou = calculate_miou(torch.from_numpy(pred_labels), torch.from_numpy(target_labels), n_classes)
        print(f"MIoU: {miou}")