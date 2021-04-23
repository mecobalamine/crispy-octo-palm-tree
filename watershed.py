import argparse
import  const
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class Watershed:
    WSHD = -1
    INQE = -2

    def __init__(self, image, kernel, levels=256):
        self.image = image
        self.kernel = kernel
        self.levels = levels

        self.markers = None
        self.next_node = None
        self.queues = None

    def generate_maskers(self):
        # Converting to grayscale
        ret, thresh = cv.threshold(self.image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        # cv.imshow("thresh", thresh)

        # Noise removal - opening morphological transformation gives in this case better results then closing.
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, self.kernel)
        # closing = cv.morphologyEx(thresh,cv.MORPH_CLOSE, self.kernel)
        # cv.imshow("opening", opening)

        # Background area determination - with dilation, to segment only background area
        background = cv.dilate(opening, self.kernel)
        # cv.imshow("background", background)

        # Foreground area determination - with erosion, to segment only foreground area
        foreground = cv.morphologyEx(thresh, cv.MORPH_ERODE, self.kernel)
        # cv.imshow("foreground", foreground)

        # Finding unknown region with background and foreground subtraction
        foreground = np.uint8(foreground)
        unknown = cv.subtract(background, foreground)
        # cv.imshow("unknown", unknown)

        # Marker labelling
        # connectedComponents() marks foreground with positive values and background with 0
        ret, markers = cv.connectedComponents(foreground)

        # Because background is now labelled with 0, which is unknown region label
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        self.markers = markers

    def ws_push(self, level, offset):
        self.next_node[offset] = -1
        tail = self.queues[level, 1]
        if tail >= 0:
            self.next_node[tail] = offset
        else:
            self.queues[level, 0] = offset
        self.queues[level, 1] = offset

    def ws_pop(self, level):
        head = self.queues[level, 0]
        _next = self.next_node[head]
        self.queues[level, 0] = _next
        if _next < 0:
            self.queues[level, 1] = -1
        self.next_node[head] = -1
        return head

    def watershed(self):
        image = self.image
        height = image.shape[0]
        width = image.shape[1]

        step = self.levels

        self.next_node = np.full((height * width), -1, int)
        self.queues = np.full((step, 2), -1, int)

        mask = self.markers

        for i in range(width):
            mask[0, i] = self.WSHD
            mask[height - 1, i] = self.WSHD

        for y in range(1, height - 1):
            mask[y, 0] = self.WSHD
            mask[y, width - 1] = self.WSHD
            for x in range(1, width - 1):
                if mask[y, x] < 0:
                    mask[y, x] = 0
                level = step + 1
                pixel = image[y, x]
                if y > 0 and mask[y - 1, x] > 0:
                    level = min(abs(int(image[y, x - 1]) - int(pixel)), level)
                if y < height and mask[y + 1, x] > 0:
                    level = min(abs(int(image[y, x - 1]) - int(pixel)), level)
                if x > 0 and mask[y, x - 1] > 0:
                    level = min(abs(int(image[y, x - 1]) - int(pixel)), level)
                if x < width and mask[y, x + 1] > 0:
                    level = min(abs(int(image[y, x - 1]) - int(pixel)), level)
                if mask[y, x] == 0 and level <= step:
                    self.ws_push(level, y * step + x)
                    mask[y, x] = self.INQE

        lv = 0
        for _ in range(step):
            if self.queues[lv, 0] >= 0:
                break
            lv += 1
        if lv == step:
            self.markers = mask
            return
        active_queue = lv

        flat_image = image.flatten()
        flat_mask = mask.flatten()

        while True:
            lab = 0

            if self.queues[active_queue, 0] < 0:
                lv = active_queue + 1
                for _ in range(active_queue + 1, step):
                    if self.queues[lv, 0] >= 0:
                        break
                    lv += 1
                if lv == step:
                    break
                active_queue = lv

            offset = self.ws_pop(active_queue)
            pixel = flat_image[offset]

            m = flat_mask[offset - 1]
            if m > 0:
                lab = m
            m = flat_mask[offset + 1]
            if m > 0:
                if lab == 0:
                    lab = m
                elif m != lab:
                    lab = self.WSHD
            m = flat_mask[offset - step]
            if m > 0:
                if lab == 0:
                    lab = m
                elif m != lab:
                    lab = self.WSHD
            m = flat_mask[offset + step]
            if m > 0:
                if lab == 0:
                    lab = m
                elif m != lab:
                    lab = self.WSHD
            flat_mask[offset] = lab
            assert (lab != 0)
            if lab == self.WSHD:
                continue

            if flat_mask[offset - 1] == 0:
                delta = abs(flat_image[offset - 1] - pixel)
                self.ws_push(delta, offset - 1)
                active_queue = min(active_queue, delta)
                flat_mask[offset - 1] = self.INQE
            if flat_mask[offset + 1] == 0:
                delta = abs(flat_image[offset + 1] - pixel)
                self.ws_push(delta, offset + 1)
                active_queue = min(active_queue, delta)
                flat_mask[offset + 1] = self.INQE
            if flat_mask[offset - step] == 0:
                delta = abs(flat_image[offset - step] - pixel)
                self.ws_push(delta, offset - step)
                active_queue = min(active_queue, delta)
                flat_mask[offset - step] = self.INQE
            if flat_mask[offset + step] == 0:
                delta = abs(flat_image[offset + step] - pixel)
                self.ws_push(delta, offset + step)
                active_queue = min(active_queue, delta)
                flat_mask[offset + step] = self.INQE

        self.markers = flat_mask.reshape(height, width)

    def run(self):
        self.generate_maskers()
        self.watershed()

        mask = self.markers
        self.image[mask == self.WSHD] = 0


def get_args():
    parser = argparse.ArgumentParser(description='Watershed Segmentation')
    parser.add_argument('--input_path', type=str, default="./assets/images",
                        help='input path')
    parser.add_argument('--output_path', type=str, default="./assets/results",
                        help='output path')
    parser.add_argument('--kernel', type=tuple, default=(5, 5),
                        help='a tuple for the Gaussian Filter')

    args = parser.parse_args()

    return args


def main(args):
    input_file = const.input_file
    output_file = const.output_file
    image = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    # Gaussian Filter
    smooth_image = cv.GaussianBlur(image, args.kernel, 0)

    w = Watershed(smooth_image, args.kernel)
    w.run()
    result = w.markers

    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image, cmap='gray')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(result, cmap='Paired', interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    main(get_args())
