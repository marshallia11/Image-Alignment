import cv2
import numpy as np
import os,io
import copy
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import tensorflow as tf


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

def unetPrepro(dirs_img,path_img,path_mask):
    img_dataset = []
    mask_dataset = []
    i=0
    for item in dirs_img:
        name, extension = os.path.splitext(item)
        if os.path.isfile(path_img+item) and  extension == '.png':
            img = cv2.imread(path_img+item,3)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = np.load(path_mask+name+'.npy')
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('mask '+item, mask)
            # cv2.imshow('img '+item, img)
            # print(mask)

            if img.shape[0] ==852:
                img = img[2:850,:]
                mask = mask[2:850,:]
                mask=np.hstack((mask, np.zeros((mask.shape[0], 1), dtype=mask.dtype)))
                img=cv2.copyMakeBorder(img.copy(), 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=0)
            if img.shape[0] ==843:
                img = img[:,1:-1]
                mask = mask[:,1:-1]
                img=cv2.copyMakeBorder(img.copy(), 2, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)
                mask=np.vstack((np.zeros((2,mask.shape[1]), dtype=mask.dtype),mask))
                mask=np.vstack((mask, np.zeros((3,mask.shape[1]), dtype=mask.dtype)))
                print(mask.shape)
            mask_dataset.append(mask)
            # print('mask_dataset ',mask_dataset.shape)
            # mask_dataset[i]=mask
            img_dataset.append(img)
            # print(img.shape)
        else:
            print(name)
        i=i+1
    return img_dataset, mask_dataset

#If you want to create your own dataset uncomment and run this file
if __name__ == '__main__':
    x_train = []
    dirs_img = os.listdir('/home/kuro/Downloads/3225_defect-free_20210218/data_annotated')
    path_img = '/home/kuro/Downloads/3225_defect-free_20210218/data_annotated/'
    # path_mask = '/home/kuro/Downloads/3225_defect-free_20210218/data_dataset_voc/SegmentationClassPNG/'
    path_mask = '/home/kuro/Downloads/3225_defect-free_20210218/data_dataset_voc/SegmentationClass/'
    (images, masks) = unetPrepro(dirs_img,path_img,path_mask)
    test =  np.vstack(masks)
    mask_dataset = test.reshape((15,848,1056,1))
    images =  np.true_divide(images, 255.0)

    # dataset = np.array(x_train)
    a =  np.array(images)
    # cv2.imshow('img', images[14])
    # cv2.imshow('mask', masks[14])
    np.save('/home/kuro/project/Image-Alignment/input/imagesDirty3.npy', a)
    np.save('/home/kuro/project/Image-Alignment/input/masksDirty3.npy', mask_dataset)
    # dataset = np.load('/home/kuro/project/Image-Alignment/input/dataset.npy',allow_pickle=True)
