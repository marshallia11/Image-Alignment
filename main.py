import util
import feature_detection as fd
import feature_matching as fm

if __name__ == '__main__':
    filepath = '/home/kuro/project/Image-Alignment/input/0122/'
    original ='1_2.png'
    wrap ='1_11.png'
    (img_ori, img_gray_ori) = util.input(filepath+original)
    (img_wrap, img_gray_wrap) = util.input(filepath+wrap)
    kp_ori, des_ori = fd.fast(img_ori, img_ori)
    kp_wrap, des_wrap = fd.fast(img_wrap, img_gray_wrap)
    fm.brute_force(img_ori, kp_ori, des_ori, img_wrap, kp_wrap, des_wrap)

