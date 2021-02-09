import cv2
import numpy as np
import matplotlib.pyplot as plt

def input(file_path):
    img = cv2.imread(file_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite('/home/kuro/project/Image-Alignment/output/img_hsv.jpg', img_gray)
    return img, img_gray;

def output(out_filepath, img):
    cv2.imwrite(out_filepath,img)
