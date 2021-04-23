import numpy as np
import scipy.io as scio
import cv2 as cv
from PIL import Image


keys = ['T00', 'T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90']
depth = 99


def read_data(path):
    pass


if __name__ == '__main__':
    ct_path = 'C:\\Users\\Ylming\\Desktop\\diploma\\dataset\\p38_4D_raw.mat'
    ct_data = scio.loadmat(ct_path)
    print(ct_data.keys())
    # for i in range(len(keys)):
    sample = ct_data.get(keys[0])
    print(sample.shape)
    mat = sample[:, :, 50]
    print(mat.shape)
    # print(mat)
    mat = cv.convertScaleAbs(mat, alpha=255.0 / np.amax(mat), beta=0)
    cv.imshow('pre_img', mat)
    cv.imwrite("./assets/seg_test.jpg", mat)
    # img = Image.fromarray(mat)
    # img.show()

    print(mat.flat[257])
    print(mat[2][1])

    target_path = 'C:\\Users\\Ylming\\Desktop\\diploma\\dataset\\p38_Formatted4DInput_segAPRIL.mat'
    target_data = scio.loadmat(target_path)
    print(target_data.keys())
    unit = target_data.get('P')
    print(unit.shape)
    sample = unit[0][0]
    print(sample.shape)
    mat = sample[:, :, 50]
    print(type(mat))
    print(mat.shape)
    # print(mat)
    mat = cv.convertScaleAbs(mat, alpha=255.0 / np.amax(mat), beta=0)
    cv.imshow('processed_img', mat)
    cv.imwrite("assets/seg_test_result.jpg", mat)

    cv.waitKey()
