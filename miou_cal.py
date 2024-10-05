import torch
import numpy as np
from PIL import Image
import os
import pprint

def fast_hist(a, b, n):
    """
    计算混淆矩阵
    a: 标签，形状为(H×W,)
    b: 预测结果，形状为(H×W,)
    n: 类别总数
    """
    k = (a >= 0) & (a < n)
    # print(np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).shape)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    """
    计算每个类别的IoU
    """
    print('Defect class IoU as follows:')
    print(np.diag(hist)[1:] / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist))[1:], 1))
    return np.diag(hist)[1:] / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist))[1:], 1)


def per_class_PA(hist):
    """
    计算每个类别的准确率
    """
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((200, 200))  # 确保图片大小为200x200
    image = np.array(image)
    return image
def convert_to_labels(image):
    # 将图片转换为标签图
    labels = np.zeros(image.shape[:2], dtype=np.int32)  # 仅包含高度和宽度
    for class_id, color in enumerate([0,1,2,3]):
        # 生成与image形状相同的mask，比较每个像素是否等于指定颜色
        # 需要添加一个维度以匹配image的形状
        # print("image.shape:",image.shape)
        mask = image == color
        # 使用mask更新labels
        labels[mask] = class_id   # 类别从1开始
    return labels


def is_npy_file(filename):
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() == '.npy'


# 假设第一张图片是类别0，第二张图片是类别1
# 这里我们使用简单的二分类示例，你需要根据实际情况调整
image1_path = '/home/SegContest/model/Pytorch-UNet/predict'  # 第一张图片的路径
image2_path = '/home/SegContest/model/NEU_Seg-main/annotations/test'  # 第二张图片的路径
pred = []
test = []
if __name__=="__main__":
    num_classes = 4
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))
    for filename in os.listdir(image1_path):
        if is_npy_file(filename):
            continue
        if os.path.isfile(os.path.join(image1_path, filename)):
            pred.append(os.path.join(image1_path, filename))
            # out_files.append(os.path.join(args.output,filename))
    for filename in os.listdir(image2_path):
        if is_npy_file(filename):
            continue
        if os.path.isfile(os.path.join(image2_path, filename)):
            test.append(os.path.join(image2_path, filename))
    Miou = 0
    ans = 0
    for i in range(len(pred)):
        # print(test[i])
        # if test[i] != "/home/SegContest/model/NEU_Seg-main/annotations/test/000321.png":
        #     continue
        filename =test[i].split('/')[-1]
        new_filename = filename.replace('.png', '.jpg')
        pred[i] = os.path.join(image1_path,new_filename)
        image1 = load_image(pred[i])
        image2 = load_image(test[i])
        
        pred_labels = convert_to_labels(image1)
        target_labels = convert_to_labels(image2)
        # print(np.array2string(pred_labels, separator=', ', prefix='[', threshold=np.inf))
        # print(np.array2string(target_labels, separator=', ', prefix='[', threshold=np.inf))
        if len(target_labels.flatten()) != len(pred_labels.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                len(target_labels.flatten()), len(pred_labels.flatten()), test[i],
                pred[i]))
            continue
        # 计算MIoU
        hist += fast_hist(target_labels.flatten(), pred_labels.flatten(), num_classes)
        hist11 = fast_hist(target_labels.flatten(), pred_labels.flatten(), num_classes)
        mIou = per_class_iu(hist11)
        mIou = round(np.nanmean(mIou), 4)
        
        if mIou < 0.5:
            print("mIou:",mIou)
            print("test:",test[i])
            print("pred:",pred[i])
            ans += 1
    mIoUs = per_class_iu(hist)
    mPA = per_class_PA(hist)
    # 输出所有类别的平均mIoU和mPA
    print("ans",ans)
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 4)) +
          '; mPA: ' + str(round(np.nanmean(mPA) * 100, 4)))