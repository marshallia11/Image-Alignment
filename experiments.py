import numpy as np
import cv2
import util
import feature_detection as fd

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    dataset = np.load('/home/kuro/project/Image-Alignment/input/dataset.npy',allow_pickle=True)
    img1 = dataset[0]
    img2 = dataset[8]
    # dataset = [dataset[1]]
    dataset = util.preprocessing(dataset)
    cnr1= fd.shi_tomasi(img1, dataset[0])
    cnr2= fd.shi_tomasi(img2, dataset[8])
    x= 0

    if(len(cnr1)< len(cnr2)):
        cnr2 =cnr2[:len(cnr1)]
    if (len(cnr2) < len(cnr1)):
        cnr1 =cnr1[:len(cnr2)]
    cv2.waitKey(0)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(dataset[0], dataset[8], cnr1, None)

    assert cnr1.shape == curr_pts.shape
    idx = np.where(status == 1)[0]
    prev_pts = cnr1[idx]
    curr_pts = curr_pts[idx]
    kp_img = cv2.drawKeypoints(img2, curr_pts, None, color=(255, 0, 0))

    M = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC, maxIters=1000,confidence=0.95)
    # Extract traslation
    # print('affine')
    # m = m[0]
    # dx = m[0][2]
    # dy = m[1][2]
    #
    # # Extract rotation angle
    # da = np.arctan2(m[1][0], m[0][0])
    # #
    # # # Store transformation
    # transform = [dx, dy, da]
    # trajectory = np.cumsum(transform, axis=0)

    # print(transform)
    # affine = m[0]
    # cv2.imshow('image2',img2)
    result = cv2.warpAffine(src=img2,M=M[0],dsize=(img2.shape[1], img2.shape[0]))
    cv2.imshow('ori',kp_img)
    cv2.imshow('wrap',img2)
    cv2.imshow('result',result)
    cv2.waitKey(0)
    # print(transform)