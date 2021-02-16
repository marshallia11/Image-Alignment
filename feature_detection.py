import cv2
import numpy as np
import matplotlib.pyplot as plt
import util

#how to make it supported
def harris_corner_detection(dataset,grayscaled):
    # img, block size, kernel size, k
    results=[]
    i =0
    for img_gray in grayscaled:
        img_gray = np.float32(img_gray)

        dst = cv2.cornerHarris(img_gray, 3,3,0.02)
        dst = cv2.dilate(dst, None)
        img = dataset[i]
        img[dst>0.01*dst.max()]=[0,0,255]
        print(len(dst))
        results.append(dst)
        i = i + 1
        # util.output('/home/kuro/project/Image-Alignment/output/haris/'+str(i)+'.png', img)
    return results

#how to make it supported
def shi_tomasi(dataset,grayscaled):
    # number of point 100, quality 0.4, min euclidean 10
    results=[]
    i = 0
    for img_gray in grayscaled:
        corners = cv2.goodFeaturesToTrack(img_gray,
                                          maxCorners=100,
                                          qualityLevel=0.3,
                                          minDistance=15,
                                          blockSize=3)
        # for i in corners:
        #     x, y = i.ravel()
        #     cv2.circle(dataset[0], (x, y), 3, 255, -1)
        print(len(corners))
        i=i+1
        results.append(corners)
    return results

def sift(img, img_gray):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    result = cv2.drawKeypoints(img_gray, kp, None, color=(255, 0, 0))
    print(len(kp))
    # util.output('/home/kuro/project/Image-Alignment/output/sift/1_2.png', result)
    return kp, des

#how to make it supported
def fast(img, img_gray):
    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(True)
    kp = fast.detect(img_gray, None)
    kp_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    print(len(kp))
    # util.output('/home/kuro/project/Image-Alignment/output/fast/1_2.png', kp_img)
    return kp

#how to make it supported
def brief(img, img_gray):
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints with STAR
    kp = star.detect(img_gray, None)
    kp, des = brief.compute(img_gray, kp)
    result = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    print(len(kp))
    return kp, des

def orb(dataset):
    i =0
    kps =[]
    desc=[]
    for img in dataset:
        orb = cv2.ORB_create(nfeatures=2000)
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        kp_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        kps.append(kp)
        desc.append(des)
        print(len(kp))
        # util.output('/home/kuro/project/Image-Alignment/output/orb/'+str(i)+'.png', kp_img)
        i=i+1
    return kps,desc

def akaze(img, img_gray):
    orb = cv2.AKAZE_create()
    kp, des = orb.detectAndCompute(img_gray, None)
    kp_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    print(len(kp))
    fig = plt.figure()
    fig.suptitle('Akaze')
    plt.axis('off')
    plt.imshow(kp_img)
    plt.show()

    # util.output('/home/kuro/project/Image-Alignment/output/orb/1_2.png', kp_img)

    return kp, des
