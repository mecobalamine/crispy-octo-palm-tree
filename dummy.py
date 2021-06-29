import cv2 as cv
import const
import numpy as np

from graph_based import Graph
from matplotlib import pyplot as plt


def merge(a, b):
    shape = a.shape
    height = shape[0]
    width = shape[1]
    res = np.ones(shape)
    for y in range(height):
        for x in range(width):
            if a[y, x] == 0 and b[y, x] == 0:
                res[y, x] = 0
    return res


def fill_hole(im_in):
    im_floodfill = im_in.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv

    return im_out


def find_dummy(image, i, j, thr, val):
    dummy = image.copy()
    stack = [(i, j)]
    while len(stack) != 0:
        seed = stack.pop()
        y, x = seed
        while y < 256 and dummy[y, x] < thr:
            dummy[y, x] = val
            y += 1
        y_right = y - 1
        y, x = seed
        y -= 1
        while 0 <= y and dummy[y, x] < thr:
            dummy[y, x] = val
            y -= 1
        y_left = y + 1
        # join left and right seeds
        if x > 0 and dummy[y_left, x - 1] < thr:
            stack.append((y_left, x - 1))
        if x < 255 and dummy[y_left, x + 1] < thr:
            stack.append((y_left, x + 1))
        if x > 0 and dummy[y_right, x - 1] < thr:
            stack.append((y_right, x - 1))
        if x < 255 and dummy[y_right, x + 1] < thr:
            stack.append((y_right, x + 1))

        # for y in range(y_left, y_right + 1):
        #
        #     if x > 0 and y > 0 and dummy[y, x - 1] < thr < dummy[y - 1, x - 1]:
        #         stack.append((y, x - 1))
        #
        #     if x < 255 and y > 0 and dummy[y, x + 1] < thr < dummy[y - 1, x + 1]:
        #         stack.append((y, x + 1))
        #
        #     if x > 0 and y > 0 and dummy[y, x - 1] > thr > dummy[y - 1, x - 1]:
        #         stack.append((y - 1, x - 1))
        #
        #     if x < 255 and y > 0 and dummy[y, x + 1] > thr > dummy[y - 1, x + 1]:
        #         stack.append((y - 1, x + 1))

    # shape = dummy.shape
    # height = shape[0]
    # width = shape[1]
    # for y in range(height):
    #     for x in range(width):
    #         if dummy[y, x] == val:
    #             dummy[y, x] = 0
    #         else:
    #             dummy[y, x] = 1
    return dummy


def find_square(dummy):
    shape = dummy.shape
    height = shape[0]
    width = shape[1]

    hor = np.zeros((height + 1, height + 1), dtype=int)
    ver = np.zeros((width + 1, width + 1), dtype=int)
    param1 = [-1, -1, 0, 0]
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            if dummy[y, x] == 1:
                continue

            left = hor[y, x - 1] + 1
            top = ver[y - 1, x] + 1
            hor[y - 1, x - 1] = left
            ver[y - 1, x - 1] = top

            if left * top > param1[2] * param1[3]:
                param1 = [y, x, left, top]

    hor = np.zeros((height + 1, height + 1), dtype=int)
    ver = np.zeros((width + 1, width + 1), dtype=int)
    param2 = [-1, -1, 0, 0]
    for y in range(height):
        for x in range(width):
            if dummy[y, x] == 1:
                continue

            right = hor[y, x + 1] + 1
            bottom = ver[y + 1, x] + 1
            hor[y + 1, x + 1] = right
            ver[y + 1, x + 1] = bottom

            if right * bottom > param2[2] * param2[3]:
                param2 = [y, x, right, bottom]

    return [param1[0], param2[0], param1[1], param2[1]]


class Dummy(object):
    debug = True
    thr = 73

    def __init__(self, image):
        self.image = image
        self.result = None
        self.dummy = None

    def run(self):
        image = self.image
        shape = image.shape
        height = shape[0]
        width = shape[1]

        # plt.hist(image.ravel(), 256, [0, 256])
        # plt.show()
        # ret, binary = cv.threshold(image, 50, 255, cv.THRESH_BINARY)  # 指定阈值50
        # print("二值阈值: %s" % ret)
        # ret_otsu, binary_otsu = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        # print("二值阈值_otsu: %s" % ret_otsu)
        # ret_tri, binary_tri = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
        # print("二值阈值_tri: %s" % ret_tri)

        result = np.zeros(shape)

        if self.debug:
            dummy = np.ones(shape)
            for y in range(height):
                for x in range(width):
                    if image[y, x] > 70:
                        # image[y, x] = 255
                        dummy[y, x] = 0

            param = find_square(dummy)
            # print(param)

            for y in range(height):
                for x in range(width):
                    if image[y, x] < 100 and not (param[0] < y < param[1] and param[2] < x < param[3]):
                        image[y, x] = self.thr
                    if image[y, x] < 100 and param[0] < y < param[1] and param[2] < x < param[3]:
                        result[y, x] = 1
                    else:
                        result[y, x] = 0
        else:
            dummy = find_dummy(image, 0, 0, 50, 255)
            dummy = find_dummy(dummy, height - 1, width - 1, 50, 255)

            for y in range(height):
                for x in range(width):
                    if dummy[y, x] != 255 and image[y, x] < 100:
                        result[y, x] = 1
                    elif dummy[y, x] == 0:
                        image[y, x] = self.thr

                    if dummy[y, x] == 255:
                        dummy[y, x] = 0
                    else:
                        dummy[y, x] = 1

            self.dummy = dummy

        # result = cv.morphologyEx(result, cv.MORPH_OPEN, const.kernel())
        # result = cv.morphologyEx(result, cv.MORPH_CLOSE, const.kernel())

        # plt.figure()
        # arr = image.flatten()
        # _ = plt.hist(arr, bins=256, range=[0, 254], facecolor='blue', alpha=0.75)
        # plt.show()

        self.image = image
        cv.imwrite('./assets/dummy.jpg', image)

        self.result = result


def main():
    input_file = const.input_file()
    image = cv.imread(input_file, cv.IMREAD_GRAYSCALE)

    dummy = Dummy(image)
    dummy.run()

    result = dummy.result

    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image, cmap='gray')
    ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(result, cmap='gray')
    # ax2.imshow(result, cmap='Paired', interpolation='nearest')
    ax2.set_axis_off()

    plt.show()


if __name__ == '__main__':
    main()
