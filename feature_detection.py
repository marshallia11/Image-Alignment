import cv2
import numpy as np
import matplotlib.pyplot as plt
import util

#how to make it supported
def harris_corner_detection(img, img_gray):
    # img, block size, kernel size, k
    img_gray = np.float32(img_gray)

    dst = cv2.cornerHarris(img_gray, 3,3,0.02)
    dst = cv2.dilate(dst, None)
    print(dst)
    img[dst>0.01*dst.max()]=[0,0,255]
    util.output('/home/kuro/project/Image-Alignment/output/haris/1_2.png', img)
    return img

#how to make it supported
def shi_tomasi(img, img_gray):
    # number of point 100, quality 0.4, min euclidean 10
    corners = cv2.goodFeaturesToTrack(img_gray, 100, 0.3, 15)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    plt.imshow(img), plt.show()
    util.output('/home/kuro/project/Image-Alignment/output/shi_tomasi/1_2.png', img)
    return img

def sift(img, img_gray):
    sift = cv2.SIFT_create()
    kp = sift.detect(img_gray, None)
    kp, des = sift.detectAndCompute(img_gray, None)
    result = cv2.drawKeypoints(img_gray, kp, None, color=(255, 0, 0))
    plt.imshow(result)
    util.output('/home/kuro/project/Image-Alignment/output/sift/1_2.png', result)
    return kp, des

#how to make it supported
def fast(img, img_gray):
    fast = cv2.FastFeatureDetector_create(11 ,True, cv2.FAST_FEATURE_DETECTOR_TYPE_5_8 )
    fast.setNonmaxSuppression(True)
    kp = fast.detect(img_gray, None)
    kp, des = fast.detectAndCompute(img_gray,None)
    kp_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    util.output('/home/kuro/project/Image-Alignment/output/fast/1_2.png', kp_img)
    return kp, des

#how to make it supported
def brief(img, img_gray):
    star = cv2.xfeatures2d.StarDetector_create()

    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints with STAR
    kp = star.detect(img_gray, None)

    # compute the descriptors with BRIEF
    kp, des = brief.compute(img_gray, kp)
    return kp, des

def orb(img, img_gray):
    orb = cv2.ORB_create(nfeatures=2000)
    kp = orb.detect(img_gray, None)
    kp, des = orb.compute(img_gray, kp)
    kp_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

    util.output('/home/kuro/project/Image-Alignment/output/orb/1_2.png', kp_img)

    return kp, des

def akaze(img, img_gray):
    orb = cv2.AKAZE_create()
    kp = orb.detect(img_gray, None)
    kp, des = orb.detectAndCompute(img_gray, None)
    kp_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

    util.output('/home/kuro/project/Image-Alignment/output/orb/1_2.png', kp_img)

    return kp, des
