import cv2
import numpy as np
import matplotlib.pyplot as plt
import util
import pandas as pd
import math

def brute_force(original, dataset, kp, desc):
    img_ori = dataset[0]
    kp_ori = kp[0]
    des_ori=desc[0]
    i = 0
    results=[]
    while i < len(dataset):
        img_wrap =dataset[i]
        ori = original[i]
        kp_wrap=kp[i]
        des_wrap=desc[i]
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        matches = matcher.match(des_ori, des_wrap)
        matches = sorted(matches, key=lambda x:x.distance)
        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")
        for (n, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images map to each other
            ptsA[n] = kp_ori[m.queryIdx].pt
            ptsB[n] = kp_wrap[m.trainIdx].pt
        result  = cv2.drawMatches(img_ori, kp_ori, img_wrap, kp_wrap, matches,None, flags=2)
        plt.imshow(result)
        plt.show()
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
        (h, w) = img_ori.shape[:2]
        aligned = cv2.warpPerspective(ori, H, (w, h))

        M = cv2.estimateAffinePartial2D(ptsA, ptsB, method=cv2.RANSAC, maxIters=1000,confidence=0.95)
        image = cv2.warpAffine(src=img_wrap,M=M[0],dsize=(img_wrap.shape[1], img_wrap.shape[0]))
        # util.output('/home/kuro/project/Image-Alignment/output/feature_matching/brute_after'+str(i)+'.png', aligned)
        # util.output('/home/kuro/project/Image-Alignment/output/feature_matching/brute_before'+str(i)+'.png', image)
        i=i+1
        results.append(image)
    return results

def knn(original,dataset, kp, desc):
    img_ori = dataset[0]
    kp_ori = kp[0]
    des_ori = desc[0]
    i = 0
    results=[]
    while i < len(dataset):
        img_wrap = dataset[i]
        ori = original[i]
        kp_wrap = kp[i]
        # print(kp_wrap)
        des_wrap = desc[i]
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des_ori, des_wrap, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good_matches.append([m])

        # Draw matches key points
        # img3 = cv2.drawMatchesKnn(img_ori, kp_ori, img_wrap, kp_wrap, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imwrite('matches.jpg', img3)

        ref_matched_kpts = np.float32([kp_ori[m[0].queryIdx].pt for m in good_matches])
        sensed_matched_kpts = np.float32([kp_wrap[m[0].trainIdx].pt for m in good_matches])
        # H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC, 5.0)

        # Transform image using prespective transformation
        # warped_image = cv2.warpPerspective(ori, H, (img_wrap.shape[1], img_wrap.shape[0]))

        # Transform image using affine transformation
        M = cv2.estimateAffinePartial2D(sensed_matched_kpts, ref_matched_kpts, method=cv2.RANSAC, maxIters=1000,confidence=0.95)
        image = cv2.warpAffine(src=img_wrap,M=M[0],dsize=(img_wrap.shape[1], img_wrap.shape[0]))

        #Display the transformed image
        # plt.imshow(img3)
        # plt.show()

        #save the image
        # util.output('/home/kuro/project/Image-Alignment/output/feature_matching/knn_after'+str(i)+'.png', warped_image)
        # util.output('/home/kuro/project/Image-Alignment/output/feature_matching/knn_before_'+str(i)+'.png', image)
        i=i+1
        results.append(image)
    return results

def lucasKanade(dataset,grayscale, kp):
    results =[]
    template = grayscale[0]
    cnr1 = kp[0]
    i=0
    for img in grayscale:
        curr_kp = kp[i]
        template_kp = kp[0]
        rgb_img = dataset[i]
        if(len(cnr1)< len(curr_kp)):
            curr_kp =curr_kp[:len(cnr1)]
        if (len(curr_kp) < len(cnr1)):
            cnr1 =cnr1[:len(curr_kp)]
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(img,template, curr_kp, None)

        assert curr_kp.shape == curr_pts.shape
        idx = np.where(status == 1)[0]
        prev_pts = curr_kp[idx]
        curr_pts = curr_pts[idx]

        M = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC, maxIters=1000,confidence=0.95)
        image = cv2.warpAffine(src=rgb_img,M=M[0],dsize=(img.shape[1], img.shape[0]))
        result = cv2.warpAffine(src=img,M=M[0],dsize=(img.shape[1], img.shape[0]))
        results.append(result)

        #Visualize the movement in template and current image
        # helper = cv2.hconcat([img,img])
        # helper = cv2.merge((helper,helper,helper))
        # for x, (new, old) in enumerate(zip(curr_pts, prev_pts)):
        #     a, b = new.ravel()
        #     print(a,b)
        #     c, d = old.ravel()
        #     print(c,d)
        #     cv2.circle(helper, (c, d), 3, 255, -2)
        #     cv2.circle(helper, (int(1055+a), b), 3, 255, -2)
        #     cv2.line(helper,(int(1055+a), b),(c,d),(255,0,0),2)

        # util.output('/home/kuro/project/Image-Alignment/output/feature_matching/lucas_matching_'+str(i)+'.png', helper)
        # util.output('/home/kuro/project/Image-Alignment/output/feature_matching/affine_after'+str(i)+'.png', image)
        # util.output('/home/kuro/project/Image-Alignment/output/affine/affine_before'+str(i)+'_1.png', image)
        i=i+1
    return results

#RMSE metrics
def calculateRMSE(original, dataset, result):
    i = 0
    before =[]
    after =[]
    template = original[0]
    while i < len(result):
        n= len(dataset[i])

        img_before = cv2.merge((dataset[i],dataset[i],dataset[i]))
        img_before[img_before[:, :, 0] > 0, 0]=255
        img_before[img_before[:, :, 0] > 0, 1]=0
        img_before[img_before[:, :, 0]> 0, 2]=0

        img_after = cv2.merge((result[i],result[i],result[i]))
        img_after[img_after[:, :, 0] > 0, 0]=255
        img_after[img_after[:, :, 0] > 0, 1]=0
        img_after[img_after[:, :, 0]> 0, 2]=0

        blendBefore = cv2.addWeighted(template, 0.5, img_before,1, 0.0)
        blendAfter = cv2.addWeighted(template, 0.5, img_after, 1, 0.0)
        scoreBefore = np.square(np.subtract(dataset[0],dataset[i])/n).mean()

        scoreAfter = np.square(np.subtract(dataset[0],result[i])/n).mean()
        after.append(math.sqrt(scoreAfter))
        before.append(math.sqrt(scoreBefore))

        # cv2.imwrite('/home/kuro/project/Image-Alignment/output/affine_before_'+str(i)+'.png',blendBefore)
        # cv2.imwrite('/home/kuro/project/Image-Alignment/output/affine_after_'+str(i)+'.png',blendAfter)

        i=i+1

    df = pd.DataFrame({'dataset': np.arange(0, 50, 1),
                       'before': before,
                       'after': after})
    cv2.waitKey(0)

    df.to_csv('/home/kuro/project/Image-Alignment/output/result_sift.csv', index=False)

#Absolute Different metrics
def calculateDiff(original, dataset, result):
    i = 0
    before =[]
    after =[]
    template = original[0]
    while i < len(result):
        img_before = cv2.merge((dataset[i],dataset[i],dataset[i]))
        img_after = cv2.merge((result[i],result[i],result[i]))
        scoreBefore = np.mean(np.abs(dataset[0] - dataset[i]))

        scoreAfter = np.mean(np.abs(result[0] - result[i]))
        after.append(scoreAfter)
        before.append(scoreBefore)
    df = pd.DataFrame({'dataset': np.arange(0, 4, 1),
                       'before': before,
                       'after': after})
    print(df)
    df.to_csv('/home/kuro/project/Image-Alignment/output/diff_brute.csv', index=False)
