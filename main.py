import util
import feature_detection as fd
import feature_matching as fm
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold

import os, sys
if __name__ == '__main__':
    original ='/home/kuro/project/Image-Alignment/input/'
    ori_filename = '1_2.png'
    wrap ='/home/kuro/project/Image-Alignment/input/'
    wrap_filename= '1_11.png'
    dataset = np.load('/home/kuro/project/Image-Alignment/input/dataset.npy',allow_pickle=True)
    grayscale_dataset = util.preprocessing(dataset)
    # grayscale_dataset = util.input_cv(dataset)
    (kp, desc) = fd.shi_tomasi(dataset, grayscale_dataset)
    fm.knn(grayscale_dataset, kp, desc)
    # fm.brute_force(grayscale_dataset, kp, desc)

