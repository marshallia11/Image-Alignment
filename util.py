import cv2
import numpy as np
import os
import copy
from skimage.color import rgb2gray

x_train = []

def input_cv(dataset):
    images =[]
    for img in dataset:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img_gray)
    return images;

def input(dataset):
    images =[]
    for img in dataset:
        gray = rgb2gray(img)
        images.append(gray)
    return np.array(images);

def change_color(dataset):
    images = []
    for img in dataset:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (threshi, img_bw) = cv2.threshold(img_gray, 0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        images.append(img_bw)
    return images

def output(out_filepath, img):
    cv2.imwrite(out_filepath,img)

def save_npy(path, dirs):
    for item in dirs:
        if os.path.isfile(path+item):
            img = cv2.imread(path+item,3)
            img = cv2.resize(img, (1055, 843))
            x_train.append(img)

def preprocessing(dataset):
    images=[]
    i=0
    for img in dataset:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blured = cv2.GaussianBlur(img_gray, (3, 3), 0)
        (threshi, img_bw) = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        thresh1 = copy.deepcopy(img_bw)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # detecting contours
        length = len(contours)
        maxArea = -1
        drawing = np.zeros(img.shape, np.uint8)
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)  # applying convex hull technique
            cv2.drawContours(drawing, [hull], 0, (0, 255, 0), 3)  # drawing convex hull
        drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
        images.append(drawing)
    return images

#If you want to create your own dataset uncomment and run this file
# if __name__ == '__main__':
#     dirs = os.listdir('/home/kuro/project/Transistor dataset/defect-free/0122/')
#     path = '/home/kuro/project/Transistor dataset/defect-free/0122/'
#     save_npy(path, dirs)
#     dataset = np.array(x_train)
#     np.save('/home/kuro/project/Image-Alignment/input/dataset.npy', dataset)
#     dataset = np.load('/home/kuro/project/Image-Alignment/input/dataset.npy',allow_pickle=True)
#     roi(dataset)