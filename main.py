import util
import feature_detection as fd
import feature_matching as fm
import numpy as np

import os, sys
if __name__ == '__main__':
    original ='/home/kuro/project/Image-Alignment/input/0122/1_2.png'
    wrap ='/home/kuro/project/Image-Alignment/input/0122/1_11.png'
    # (img_ori, img_gray_ori) = util.input(original)
    # (img_wrap, img_gray_wrap) = util.input(wrap)
    (img_ori, img_gray_ori) = util.change_color(original)
    (img_wrap, img_gray_wrap) = util.change_color(wrap)
    fd.shi_tomasi(img_wrap, img_gray_wrap)
    kp_ori, des_ori = fd.orb(img_ori, img_ori)
    kp_wrap, des_wrap = fd.orb(img_wrap, img_gray_wrap)
    fm.brute_force(img_gray_ori, kp_ori, des_ori, img_gray_wrap, kp_wrap, des_wrap)
    # fm.knn(img_gray_ori, kp_ori, des_ori, img_gray_wrap, kp_wrap, des_wrap)

