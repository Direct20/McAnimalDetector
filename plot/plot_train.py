import matplotlib.pyplot as plt
import pandas as pd

# 读取result.csv文件
df = pd.read_csv('./ result.csv')

# 1. 绘制损失值随训练轮次变化曲线
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['loss'], label='Loss', marker='o', color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Training Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 2. 绘制平均精度均值（mAP）随训练轮次变化曲线（如果有该列）
if'mAP' in df.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['mAP'], label='mAP', marker='s', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP over Training Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# 3. 绘制各类别准确率随训练轮次变化曲线（以下示例假设有两个类别class1和class2，按需扩展）
class_columns = [col for col in df.columns if 'class' in col and 'accuracy' in col]
for class_col in class_columns:
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df[class_col], label=class_col, marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{class_col} over Training Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# 4. 可以添加更多自定义可视化，比如不同指标之间的对比等
# 例如绘制同一轮次下损失值和mAP的对比（如果有mAP列）
if'mAP' in df.columns:
    plt.figure(figsize=(10, 6))
    epochs = df['epoch']
    plt.plot(epochs, df['loss'], label='Loss', marker='o', color='r')
    plt.plot(epochs, df['mAP'], label='mAP', marker='s', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Loss vs mAP over Training Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()