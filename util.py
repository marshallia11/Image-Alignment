import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from PIL import Image
import copy
from skimage.color import rgb2gray
from skimage.filters import threshold_yen, threshold_otsu, threshold_triangle

x_train = []
dirs = os.listdir('/home/kuro/project/Transistor dataset/defect-free/0122/')
path = '/home/kuro/project/Transistor dataset/defect-free/0122/'

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
        # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # ax = axes.ravel()
        #
        # ax[0].imshow(img)
        # ax[0].set_title("Original")
        # ax[1].imshow(gray, cmap=plt.cm.gray)
        # ax[1].set_title("Grayscale")
        #
        # fig.tight_layout()
        # plt.show()
    return np.array(images);

def change_color(dataset):
    images = []
    for img in dataset:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (threshi, img_bw) = cv2.threshold(img_gray, 0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        images.append(img_bw)
    return images

def threshold_yen(img):
    threshold = threshold_yen(img)
    img_bw = img > threshold
    ax = plt.axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].hist(img_bw, cmap=plt.cm.gray)
    ax[1].set_title('thresholded')
    ax[1].axis('off')
    plt.show()
    return img_bw

def output(out_filepath, img):
    cv2.imwrite(out_filepath,img)

def save_npy(path, dirs):
    for item in dirs:
        if os.path.isfile(path+item):
            img = cv2.imread(path+item,3)
            img = cv2.resize(img, (1055, 843))
            x_train.append(img)

def roi(dataset):
    fromCenter = False
    img = dataset[0]
    r = cv2.selectROI('Image', img, fromCenter)
    imCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    cv2.imshow('crop', imCrop)
    cv2.imshow('ori', img)
    cv2.waitKey(0)

def preprocessing(dataset):
    images=[]
    i=0
    for img in dataset:
        # cv2.imshow('image', img_gray)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        blured = cv2.GaussianBlur(img_gray, (3, 3), 0)
        # cv2.imshow('blured', img_gray)
        (threshi, img_bw) = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv2.imshow('blacked', img_bw)

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
            # cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)  # drawing contours
            # rect = cv2.minAreaRect(res)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            cv2.drawContours(drawing, [hull], 0, (0, 255, 0), 3)  # drawing convex hull
            # cv2.drawContours(img, [box], -1, (0, 0, 255), 2)
        # cv2.imshow('contour', drawing)
        # cv2.waitKey(0)
        drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
        images.append(drawing)
        # output('/home/kuro/project/Image-Alignment/output/'+str(i)+'.png',img)
        # output('/home/kuro/project/Image-Alignment/output/'+str(i)+'_1.png',drawing)
        i=i+1
    return images


# if __name__ == '__main__':
#     save_npy(path, dirs)
#     dataset = np.array(x_train)
#     np.save('/home/kuro/project/Image-Alignment/input/dataset.npy', dataset)
    # dataset = np.load('/home/kuro/project/Image-Alignment/input/dataset.npy',allow_pickle=True)
    # roi(dataset)