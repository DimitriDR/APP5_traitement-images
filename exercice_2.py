import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename: str = "four.png"

    opencv_image = cv2.imread(filename)

    # 1. Filtrage gaussien
    blur_kernel = (3, 3)
    blur_sigma_X = 0  # 0 = auto
    gaussian_filtered_image = cv2.GaussianBlur(opencv_image, blur_kernel, blur_sigma_X)
