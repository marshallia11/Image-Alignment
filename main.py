import util
import feature_detection as fd
import feature_matching as fm
import numpy as np
import cv2

import matplotlib.pyplot as plt


def helper(x):
    kp =[]
    # print(x[0].reshape((1,2)))
    for v in x:
        kp.append(v.reshape((1,2)))
    return np.array(kp)
if __name__ == '__main__':
    dataset = np.load('/Users/niyanqin/PycharmProjects/LearningDL/dataset/semantic_dataset/output/predictd3.npy',allow_pickle=True)
    # dataset = np.load('/Users/niyanqin/PycharmProjects/LearningDL/dataset/semantic_dataset/input/imagesDirty3.npy',allow_pickle=True)
    # dataset = np.load('/Users/niyanqin/PycharmProjects/LearningDL/dataset/semantic_dataset/input/semantic.npy',allow_pickle=True)
    # dataset = np.load('/Users/niyanqin/PycharmProjects/LearningDL/dataset/semantic_dataset/input/masksDirty_local.npy',allow_pickle=True)

    #########Preproces the image you can choose either only grayscale usng input_cv #########
    #########or use complex preprocessing such as contour using preprocessing method#########

    # img_gray = cv2.cvtColor(dataset[0], cv2.COLOR_RGB2GRAY)

    print(dataset[0].shape)

    grayscale_dataset = util.preprocessing(dataset)
    # grayscale_dataset = util.input_cv(dataset)

    #########Select the feature detection you want to use#########
    # (kp, desc) = fd.orb(dataset)




    (kp, desc) = fd.orb(grayscale_dataset[0:15])
    # kp = fd.fast(dataset[0:100],grayscale_dataset)
    # (kp,desc) = fd.sift(dataset[0:15],dataset)

    print(kp)
    # (kp,desc) = fd.brief(dataset[0:100],grayscale_dataset)
    # (kp,desc) = fd.akaze(dataset[0:100],grayscale_dataset)
    kp = [cv2.KeyPoint_convert(x) for x in kp]
    kp = [helper(x) for x in kp]
    # print(kps[0].shape)
    # print(kps[0][0])
    # kp = fd.harris_corner_detection(dataset[0:100],grayscale_dataset)
    # kp = fd.shi_tomasi(dataset[0:100],grayscale_dataset)
    # print(kp[0].shape)
    # print(type(kp[0][0]))

    #########Select the feature matching you want to use#########

    plt.imshow(dataset[0])
    plt.show()

    results = fm.lucasKanade(dataset[0:15],grayscale_dataset[0:15], kp)
    # results = fm.brute_force(dataset[0:4],grayscale_dataset, kp, desc)
    # results = fm.knn(dataset[0:4],grayscale_dataset, kp, desc)
    #
    #########Select the matrics you want to use#########
    fm.calculateRMSE(dataset[0:15],grayscale_dataset[0:15], results)
    # fm.calculateDiff(dataset[0:4],grayscale_dataset[0:4], results[0:4])

