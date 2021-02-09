import cv2
import matplotlib.pyplot as plt

def brute_force(img_ori, kp_ori, des_ori, img_wrap, kp_wrap, des_wrap):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = matcher.match(des_ori, des_wrap)
    matches = sorted(matches, key=lambda x:x.distance)
    result  = cv2.drawMatches(img_ori, kp_ori, img_wrap, kp_wrap, matches,None, flags=2)
    plt.imshow(result)
    plt.show()
