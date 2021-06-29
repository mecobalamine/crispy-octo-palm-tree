import numpy as np
import scipy.io as scio
import cv2 as cv
from PIL import Image

root = 'C:\\Users\\Ylming\\Desktop\\diploma\\dataset'

keys = ['T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90']
depth = 99


def read_data():
    # for i in range(len(keys)):
    sample = src_data.get(keys[3])
    print(sample.shape)

    _max = 0
    for i in range(depth):
        mat = sample[:, :, i]
        _max = max(_max, np.amax(mat))
        # print(np.amax(mat))
        # print(type(mat))
        # print(mat.shape)
        # print(mat)
        mat = cv.convertScaleAbs(mat, alpha=255.0 / 2096.0, beta=0)
        # mat = convert(mat)
        # cv.imshow('pre_img', mat)
        cv.imwrite('./assets/origin1/' + str(i) + '.jpg', mat)
        # img = Image.fromarray(mat)
        # img.show()

    print(_max)
    _max = 0

    rst_path = 'C:\\Users\\Ylming\\Desktop\\diploma\\dataset\\p38_Formatted4DInput_segAPRIL.mat'
    rst_data = scio.loadmat(rst_path)
    print(rst_data.keys())
    unit = rst_data.get('P')
    print(unit.shape)
    sample = unit[0][3]
    print(sample.shape)
    for i in range(depth):
        mat = sample[:, :, i]
        _max = max(_max, np.amax(mat))
        # print(type(mat))
        # print(mat.shape)
        # print(mat)
        mat = cv.convertScaleAbs(mat, alpha=255.0 / 1096.0, beta=0)
        # cv.imshow('processed_img', mat)
        cv.imwrite('./assets/terminal1/' + str(i) + '.jpg', mat)

    print(_max)


def convert(mat):
    shape = mat.shape
    height = shape[0]
    width = shape[1]
    _max = np.max(mat)
    _min = np.min(mat)
    print(_min)
    print(_max)
    res = np.zeros(shape)
    for y in range(height):
        for x in range(width):
            res[y, x] = 256.0 * (mat[y, x] - _min) / (mat[y, x] - _max) + abs(_min)
    return res


if __name__ == '__main__':
    src_path = 'C:\\Users\\Ylming\\Desktop\\diploma\\dataset\\p38_4D_raw.mat'
    src_data = scio.loadmat(src_path)
    print(src_data.keys())

    read_data()
