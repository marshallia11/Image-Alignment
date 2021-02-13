import cv2
import numpy as np
import matplotlib.pyplot as plt
import util

def brute_force(img_ori, kp_ori, des_ori, img_wrap, kp_wrap, des_wrap):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = matcher.match(des_ori, des_wrap)
    matches = sorted(matches, key=lambda x:x.distance)
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kp_ori[m.queryIdx].pt
        ptsB[i] = kp_wrap[m.trainIdx].pt
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # print(H)
    # print(mask)
    (h, w) = img_ori.shape[:2]
    aligned = cv2.warpPerspective(img_wrap, H, (w, h))

    result  = cv2.drawMatches(img_ori, kp_ori, img_wrap, kp_wrap, matches,None, flags=2)
    util.output('/home/kuro/project/Image-Alignment/output/feature_matching/align_image_4000.png', aligned)
    util.output('/home/kuro/project/Image-Alignment/output/feature_matching/feature_matching_4000.png', result)

    plt.imshow(result)
    plt.show()

def knn(img_ori, kp_ori, des_ori, img_wrap, kp_wrap, des_wrap):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des_ori, des_wrap, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # Draw matches
    img3 = cv2.drawMatchesKnn(img_ori, kp_ori, img_wrap, kp_wrap, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('matches.jpg', img3)
    ref_matched_kpts = np.float32([kp_ori[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp_wrap[m[0].trainIdx].pt for m in good_matches])
    H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC, 5.0)

    # Warp image
    # warped_image = cv2.warpPerspective(img_wrap, H, (img_wrap.shape[1], img_wrap.shape[0]))

    plt.imshow(img3)
    plt.show()
    cv2.imwrite('warped.jpg', img3)