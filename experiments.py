import numpy as np
import cv2
import util
import feature_detection as fd

if __name__ == '__main__':
    dataset = np.load('/home/kuro/project/Image-Alignment/input/dataset.npy',allow_pickle=True)
    img1 = dataset[0]
    img2 = dataset[8]
    # dataset = [dataset[1]]
    dataset = util.preprocessing(dataset)
    cnr1= fd.shi_tomasi(img1, dataset[0])
    cnr2= fd.shi_tomasi(img2, dataset[8])
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(dataset[0],  dataset[8], cnr1, None)
    # assert cnr1.shape == cnr2.shape
