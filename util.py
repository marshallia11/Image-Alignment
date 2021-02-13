import cv2
import numpy as np
import time
import os, sys
from PIL import Image
import matplotlib.pyplot as plt

x_train = []


def input(file_path):
    img = cv2.imread(file_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray;

def change_color(file_path):
    img = cv2.imread(file_path)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hue, saturation, value, = cv2.split(img_hsv)  # ---Splitting HSV image to 3 channels---
    # ret, th = cv2.threshold(hue, 38, 255, 0)
    # lower_blue = np.array([24,100,200])
    # upper_blue = np.array([36,150,200])
    # # Threshold the HSV image to get only blue colors
    # mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',img_gray)


    (threshi, img_bw) = cv2.threshold(img_gray, 200,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    plt.imshow(img_bw)
    plt.show()
    return img, img_bw

def output(out_filepath, img):
    cv2.imwrite(out_filepath,img)

def save_npy(path, dirs):
    for item in dirs:
        if os.path.isfile(path+item):
            print(item)
            im =Image.open(path+item).convert("RGB")
            im = np.array(im)
            x_train.append(im)

# if __name__ == '__main__':
#     save_npy(path, dirs)
#     dataset = np.array(x_train)
#     np.save('/home/kuro/project/Image-Alignment/input/0122/dataset.npy', dataset)