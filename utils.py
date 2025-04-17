import cv2
from PyQt5.QtGui import QImage
from PyQt5 import QtCore
import numpy as np


def np2cv2(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def cv22np(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def cv2QImage(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w

    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line,
                                  QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(551, 361, QtCore.Qt.KeepAspectRatio)
    return convert_to_Qt_format


def np2QImage(np_img):
    if np_img.dtype != np.uint8:
        raise ValueError("NumPy array must be of type uint8.")

    if len(np_img.shape) == 2:  # 灰度图像
        height, width = np_img.shape
        channels = 1
        format = QImage.Format_Grayscale8
    elif len(np_img.shape) == 3:  # 彩色图像
        height, width, channels = np_img.shape
        if channels == 3:
            format = QImage.Format_RGB888
        elif channels == 4:
            format = QImage.Format_ARGB32
        else:
            raise ValueError(
                "NumPy array must have 3 or 4 channels for color images.")
    else:
        raise ValueError("Unsupported NumPy array shape.")

    # 使用 tobytes() 将数据转换为字节序列
    bytes_per_line = channels * width
    qimage = QImage(np_img.tobytes(), width, height, bytes_per_line, format)
    return qimage


import torch


def tensor2np(image: torch.Tensor):
    image = image.cpu().squeeze(0)
    image: np.ndarray = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image + 1
    image = image * 127.5
    image = image.astype(np.uint8)
    return image


def np2tensor():
    pass


from PyQt5 import QtGui


def np2QPixmap(image):
    image = np2QImage(image)
    pixmap = QtGui.QPixmap.fromImage(image)
    return pixmap


from PlotDialog import PlotDialog


def showPlotDialog(dialogObject: PlotDialog, title, image, modal=False):
    dialogObject.setWindowTitle(title)
    dialogObject.plot(image)
    dialogObject.setModal(modal)
    dialogObject.show()


def createPlotDialog(dialogClass: PlotDialog, title, modal=False):
    dialog = dialogClass()
    dialog.setWindowTitle(title)
    dialog.setModal(modal)
    return dialog


def hidePlotDialog(dialogObject: PlotDialog):
    dialogObject.hide()


# def apply_style_sheet(widget, style_sheet_path='./theme/modern/qss/blacksoft.css'):
#     # with open(style_sheet_path, 'r', encoding='UTF-8') as file:
#     #     style_sheet = file.read()
#     #     widget.setStyleSheet(style_sheet)
#     pass