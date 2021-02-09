import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner_detection(img, img_gray):
    # img, block size, kernel size, k
    img_gray = np.float32(img_gray)

    dst = cv2.cornerHarris(img_gray, 3,3,0.02)
    dst = cv2.dilate(dst, None)
    img[dst>0.01*dst.max()]=[0,0,255]
    return img

def shi_tomasi(img, img_gray):
    # number of point 100, quality 0.4, min euclidean 10
    corners = cv2.goodFeaturesToTrack(img_gray, 100, 0.3, 15)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    plt.imshow(img), plt.show()
    return img

def sift(img, img_gray):
    print('sift')

def surf(img, img_gray):
    print('surf')

def fast(img, img_gray):
    fast = cv2.FastFeatureDetector()
    kp = fast.detect(img_gray, None)
    img2 = cv2.drawKeypoints(img, kp, color=(255, 0, 0))

    # Print all default params
    # print
    # "Threshold: ", fast.getInt('threshold')
    # print
    # "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
    # print
    # "neighborhood: ", fast.getInt('type')
    # print
    # "Total Keypoints with nonmaxSuppression: ", len(kp)
    #
    # cv2.imwrite('fast_true.png', img2)
    #
    # # Disable nonmaxSuppression
    # fast.setBool('nonmaxSuppression', 0)
    # kp = fast.detect(img, None)
    #
    # print
    # "Total Keypoints without nonmaxSuppression: ", len(kp)
    #
    # img3 = cv2.drawKeypoints(img, kp, color=(255, 0, 0))
    #
    plt.imshow(img3)
    return img2


def brief(img, img_gray):
    print('brief')

def orb(img, img_gray):
    print('orb')

def ransac(img, img_gray):
    print('ransac')
