import util
import feature_detection as fd

if __name__ == '__main__':
    filename ='1_2.png'
    filepath = '/home/kuro/project/Image-Alignment/input/0122/'
    (img, img_gray) = util.input(filepath+filename)
    result = fd.fast(img, img_gray)
    # util.output('/home/kuro/project/Image-Alignment/output/fast/1_2.png', result)

