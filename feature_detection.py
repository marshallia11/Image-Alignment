import cv2
import numpy as np

def harris_corner_detection(dataset,grayscaled):
    results=[]
    i =0
    for img_gray in grayscaled:
        img_gray = np.float32(img_gray)

        dst = cv2.cornerHarris(img_gray, blockSize=3, ksize= 3, k=0.02)
        dst = cv2.dilate(dst, None)
        img = dataset[i]
        img[dst>0.01*dst.max()]=[0,0,255]
        results.append(dst)
        i = i + 1
    return results

def shi_tomasi(dataset,grayscaled):
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
        i=i+1
        results.append(corners)
    return results

# this method is only for one image not whole dataset
def sift(img, img_gray):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    # result = cv2.drawKeypoints(img_gray, kp, None, color=(255, 0, 0))
    # util.output('/home/kuro/project/Image-Alignment/output/sift/1_2.png', result)
    return kp, des

# this method is only for one image not whole dataset
def fast(img, img_gray):
    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(True)
    kp = fast.detect(img_gray, None)
    kp_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    # util.output('/home/kuro/project/Image-Alignment/output/fast/1_2.png', kp_img)
    return kp

# this method is only for one image not whole dataset
def brief(img, img_gray):
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    kp = star.detect(img_gray, None)
    kp, des = brief.compute(img_gray, kp)
    # result = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    return kp, des

def orb(dataset):
    i =0
    kps =[]
    desc=[]
    for img in dataset:
        orb = cv2.ORB_create(nfeatures=2000)
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        kps.append(kp)
        desc.append(des)
        # kp_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        # util.output('/home/kuro/project/Image-Alignment/output/orb/'+str(i)+'.png', kp_img)
        i=i+1
    return kps,desc

def akaze(img, img_gray):
    orb = cv2.AKAZE_create()
    kp, des = orb.detectAndCompute(img_gray, None)
    # kp_img = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    # fig = plt.figure()
    # fig.suptitle('Akaze')
    # plt.axis('off')
    # plt.imshow(kp_img)
    # plt.show()

    # util.output('/home/kuro/project/Image-Alignment/output/orb/1_2.png', kp_img)

    return kp, des
