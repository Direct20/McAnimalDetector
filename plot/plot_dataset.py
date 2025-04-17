import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 定义类别名称列表
CLASSES_NAME = [
    'Creeper', 'Enderman', 'Fox', 'Mooshroom', 'Ocelot', 'Panda', 'Polar Bear',
    'Skeleton', 'Slime', 'Spider', 'Witch', 'Wither Skeleton', 'Wolf',
    'Zombie', 'Zombified Piglin', 'Bee', 'Chicken', 'Cow', 'Iron Golem', 'Pig'
]
CLASS_PALETTE = [
    (255, 0, 0),  # 红色
    (0, 255, 0),  # 绿色
    (0, 0, 255),  # 蓝色
    (255, 255, 0),  # 黄色
    (255, 0, 255),  # 品红
    (0, 255, 255),  # 青色
    (128, 0, 0),  # 暗红色
    (0, 128, 0),  # 暗绿色
    (0, 0, 128),  # 深蓝色
    (128, 128, 0),  # 暗黄色
    (128, 0, 128),  # 暗品红
    (0, 128, 128),  # 暗青色
    (255, 165, 0),  # 橙色
    (139, 69, 19),  # 棕色
    (255, 192, 203),  # 粉红色
    (102, 205, 170),  # 薄荷绿
    (173, 216, 230),  # 浅蓝色
    (240, 230, 140),  # 卡其色
    (218, 165, 32),  # 金色
    (148, 0, 211),  # 深紫色
]


def main():

    # 定义训练集图片所在的文件夹路径，请替换成你实际的路径
    train_images_folder = "./datasets/mob.v4i.yolov11/train/images"

    # 获取训练集图片文件列表
    image_files = [
        os.path.join(train_images_folder, file)
        for file in os.listdir(train_images_folder) if file.endswith('.jpg')
    ]

    for image_file in image_files:
        # 使用cv2读取图片（方便后续绘制标注框等操作，matplotlib也可读取但绘制操作稍复杂些）
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换色彩空间以适配matplotlib显示

        # 获取对应的标注文件路径（假设标注文件和图片文件同名只是后缀为.txt）
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_file = label_file.replace('images', 'labels')
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # 计算标注框的坐标（左上角和右下角坐标）
                    h, w, _ = image.shape
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)

                    class_name = CLASSES_NAME[class_id]
                    class_color = CLASS_PALETTE[class_id]
                    # 在图片上绘制标注框（这里简单用红色矩形表示，你可根据实际情况调整颜色等样式）
                    cv2.rectangle(image, (x1, y1), (x2, y2), class_color, 2)


                    # 计算文字底纹的坐标和尺寸（这里底纹矩形的宽高根据文字长度和字体大小等估算）
                    (text_w,
                     text_h), _ = cv2.getTextSize(class_name,
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.5, 2)
                    text_bg_x1 = x1
                    text_bg_y1 = y1 - text_h - 5
                    text_bg_x2 = x1 + text_w
                    text_bg_y2 = y1 - 5
                    # 绘制文字底纹（实心矩形，颜色可根据喜好调整，这里用浅灰色）
                    cv2.rectangle(image, (text_bg_x1, text_bg_y1),
                                  (text_bg_x2, text_bg_y2), class_color, -1)
                    # 在底纹上绘制文字
                    cv2.putText(image, class_name, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                tuple((-np.array(class_color)).tolist()), 2)

        plt.imshow(image)
        # plt.title("Training Image")
        plt.axis('off')
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
