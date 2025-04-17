import cv2
from PIL import Image
import numpy as np


import numpy as np

import cv2
import numpy as np


def image_gray(image: np.ndarray, method='weighted'):
    """gray the image

    Args:
        image (np.ndarray): Image.
        method (str, optional): Method in [weighted,max,average]. Defaults to 'weighted'.
    Returns:
        image (np.ndarray)
    """
    if method == 'weighted':
        # 使用cv2的加权平均灰度转换方法，它直接返回单通道灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 通过cv2的merge函数将单通道灰度图扩展为3通道，每个通道值相同
        image_np_gray = cv2.merge([gray_image, gray_image, gray_image])
    elif method == 'max':
        # 先分离图像的BGR三个通道
        b, g, r = cv2.split(image)
        # 计算每个像素位置上BGR三个通道的最大值，得到单通道的最大值灰度图
        max_image = np.maximum.reduce([b, g, r])
        # 将单通道最大值灰度图扩展为3通道，每个通道值相同
        image_np_gray = cv2.merge([max_image, max_image, max_image])
    elif method == 'average':
        # 先分离图像的BGR三个通道
        b, g, r = cv2.split(image)
        # 计算每个像素位置上BGR三个通道的平均值，得到单通道的平均灰度图
        avg_image = (b.astype(np.float32) + g.astype(np.float32) + r.astype(np.float32)) / 3
        avg_image = avg_image.astype(np.uint8)
        # 将单通道平均灰度图扩展为3通道，每个通道值相同
        image_np_gray = cv2.merge([avg_image, avg_image, avg_image])
    else:
        raise NotImplementedError(f'{method} is not implemented')

    return image_np_gray


 
def image_binarize(image: np.ndarray,
                   threshold=127,
                   maxval=255,
                   method=cv2.THRESH_OTSU):
    """
    Binarize an image. If the image is color, it will be converted to grayscale first.
 
    Args:
        image (np.ndarray): Input image. Can be grayscale or color.
        threshold (int, optional): Threshold value. Defaults to 127.
        maxval (int, optional): Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types. Defaults to 255.
        method (int, optional): Thresholding method. Defaults to cv2.THRESH_OTSU.
 
    Returns:
        np.ndarray: Binarized image.
    """
    # Check if the image is color (3 channels)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert color image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # If it's already grayscale or single channel, use it directly
        gray_image = image
 
    # Apply thresholding
    ret, binarized_image = cv2.threshold(gray_image, threshold, maxval, method)
    
    return binarized_image


def image_equalize_hist(image: np.ndarray, method='normal', **kwargs):
    """
    Equalize histogram for a grayscale or color image.
 
    Args:
        image (np.ndarray): Image. Should be a 2D array for grayscale or a 3D array for color images.
        method (str, optional): Method in ['normal', 'adaptive']. Defaults to 'normal'.
 
    Returns:
        np.ndarray: Image with equalized histogram.
    """
    if len(image.shape) == 2:  # Grayscale image
        if method == 'normal':
            return cv2.equalizeHist(image)
        elif method == 'adaptive':
            clahe = cv2.createCLAHE(clipLimit=kwargs.get('clip_limit', 2),
                                    tileGridSize=kwargs.get(
                                        'tile_grid_size', (8, 8)))
            return clahe.apply(image)
    elif len(image.shape) == 3 and image.shape[2] == 3:  # Color image
        b_channel, g_channel, r_channel = cv2.split(image)

        if method == 'normal':
            b_channel_eq = cv2.equalizeHist(b_channel)
            g_channel_eq = cv2.equalizeHist(g_channel)
            r_channel_eq = cv2.equalizeHist(r_channel)
        elif method == 'adaptive':
            clahe = cv2.createCLAHE(clipLimit=kwargs.get('clip_limit', 2),
                                    tileGridSize=kwargs.get(
                                        'tile_grid_size', (8, 8)))
            b_channel_eq = clahe.apply(b_channel)
            g_channel_eq = clahe.apply(g_channel)
            r_channel_eq = clahe.apply(r_channel)
        else:
            raise NotImplementedError(
                "Method '{}' is not implemented.".format(method))

        # Merge the channels back
        equ_hist_image = cv2.merge((b_channel_eq, g_channel_eq, r_channel_eq))
        return equ_hist_image
    else:
        raise ValueError(
            "Input image must be either a 2D grayscale image or a 3D color image with 3 channels."
        )


def image_histogram(image: np.ndarray, color: tuple):
    """Generate histogram image.

    Args:
        image (np.ndarray): Image.
        color (tuple): (R,G,B)

    Returns:
        np.ndarray: Histogram
    """
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)

    return histImg


def image_edge_detect(image: np.ndarray, method='canny', **kwargs):
    edge_image = None
    if method == 'canny':
        edge_image = cv2.Canny(image, 15, 35)  # kwargs['t1'],
        #kwargs['t2'])  # canny边缘检测 15, 35
    elif method == 'laplacian':
        edge_image = cv2.Laplacian(image, -1)  # 拉普拉斯变换
    elif method == 'sobel':
        edge_image = cv2.Sobel(image, -1, 1, 0, ksize=3)  # sobel变换
    else:
        raise NotImplementedError()
    return edge_image
