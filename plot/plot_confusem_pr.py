import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
import numpy as np
import cv2
from torch.utils.data import DataLoader
from net.mobilev1 import MobileNetV1
from net.mobilev1_leakyrelu import MobileNetV1WithLeakyReLU
from sklearn.metrics import precision_recall_curve, average_precision_score
from PIL import Image
# CIFAR-10 类别名
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]
from img_proc import *


class HistogramEqualizationTransform:

    def histogram_equalization(self, image):
        image = np.array(image)
        image = image_gray(image)
        image = image_equalize_hist(image, method='adaptive')
        image = np.expand_dims(image, axis=2)
        image = image.repeat(3, axis=2)
        return image

    def __call__(self, img):
        img = self.histogram_equalization(img)
        return Image.fromarray(img)


# 加载CIFAR10数据集
transform = transforms.Compose([
    # HistogramEqualizationTransform(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10测试集
test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载已经训练好的MobileNetV1模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV1WithLeakyReLU(num_classes=10)
model.load_state_dict(
    torch.load(
        './work_dirs/mobilenetv1/MobileNetV1_epoch7.pth'
    ))  # 加载模型权重
model.to(device)
model.eval()

# 计算测试集精度、混淆矩阵、F1score、PR曲线
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 计算测试精度
accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=CIFAR10_CLASSES,
            yticklabels=CIFAR10_CLASSES)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# 计算F1score
f1 = f1_score(y_true, y_pred, average='weighted')
print(f'F1 Score: {f1:.4f}')

# 二值化真实标签
y_true_bin = np.array([[1 if label == i else 0 for label in y_true]
                       for i in range(10)])  # 每个类别的二值化真实标签
y_scores = []  # 每个类别的预测概率

# 获取模型对每个类别的预测概率
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs).softmax(dim=1).cpu().numpy()
        y_scores.append(outputs)

y_scores = np.vstack(y_scores)

# 计算PR曲线和平均精度
precision = {}
recall = {}
average_precision = {}

for i in range(10):
    precision[i], recall[i], _ = precision_recall_curve(
        y_true_bin[i], y_scores[:, i])
    average_precision[i] = average_precision_score(y_true_bin[i], y_scores[:,
                                                                           i])

# 绘制多分类PR曲线
plt.figure(figsize=(10, 7))
for i, class_name in enumerate(CIFAR10_CLASSES):
    plt.plot(recall[i],
             precision[i],
             lw=2,
             label=f'{class_name} (area = {average_precision[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()
