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
        i=i+1
        plt.show()
        util.output('/home/kuro/project/Image-Alignment/output/haris/'+str(i)+'.png', img)
    return results

#how to make it supported
def shi_tomasi(dataset,grayscaled):
    # number of point 100, quality 0.4, min euclidean 10
    i =0
    results=[]
    # for img_gray in grayscaled:
    print(i)
    corners = cv2.goodFeaturesToTrack(grayscaled, 100, 0.3, 15, 3)
    # corners = np.int0(corners)
    img = dataset[i]
    i=i+1
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)
    results.append(corners)
    plt.imshow(img)
    plt.show()

    # util.output('/home/kuro/project/Image-Alignment/output/shitomsi/' + str(i) + '.png', img)
    # return results
    return corners
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
        util.output('/home/kuro/project/Image-Alignment/output/orb/'+str(i)+'.png', kp_img)
        i=i+1
    return kps,desc

def akaze(img, img_gray):
    orb = cv2.AKAZE_create()
    kp = orb.detect(img_gray, None)
    kp, des = orb.detectAndCompute(img_gray, None)
    kp_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

    util.output('/home/kuro/project/Image-Alignment/output/orb/1_2.png', kp_img)

    return kp, des
