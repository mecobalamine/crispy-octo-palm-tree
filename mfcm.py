import argparse
import const
import cv2 as cv
import numpy as np
import random
import sys

from matplotlib import pyplot as plt


def calculate_distance_window(win_size, win_center):
    center_y = win_center[1]
    center_x = win_center[0]

    dist_window = np.zeros(win_size)
    for y in range(win_size[1]):
        for x in range(win_size[0]):
            dist_window[y][x] = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)  # Euclidean distance
    return dist_window


def find_reliable_set(window, win_size, win_center):
    mean_gray = np.mean(window)
    median_gray = np.median(window)
    center_gray = window[win_center[0], win_center[1]]

    mean_window = np.full((win_size[1], win_size[0]), mean_gray)
    median_window = np.full((win_size[1], win_size[0]), median_gray)
    center_window = np.full((win_size[1], win_size[0]), center_gray)

    delta = window - center_window  # neighbour - med

    deviation = np.sqrt((np.mean(pow(delta, 2))))

    reliable_mat = (delta <= deviation).astype(int)  # 1: reliable, 0: unreliable
    return reliable_mat


def sliding_window(image, neighbour_effect, win_size):
    # slide a window across the image
    win_center = (int(win_size[1] / 2), int(win_size[0] / 2))  # center coordination of the window
    dist_window = calculate_distance_window(win_size, win_center)
    # Euclidean distance matrix between center coordination and neighbor coordination

    filtered_image = []

    for y in range(image.shape[0] - int(win_size[0] / 2)):
        for x in range(image.shape[1] - int(win_size[1] / 2)):
            cur_window = image[y:y + win_size[1], x:x + win_size[0]]  # neighbour window

            if cur_window.shape[0] < win_size[1] or cur_window.shape[1] < win_size[0]:
                continue

            if np.count_nonzero(cur_window) == 0:  # Pass if all values in window are zero
                filtered_image.append(0)
                continue

            # --------------Find reliable set matrix--------------
            reliable_mat = find_reliable_set(cur_window, win_size, win_center)

            # --------------Weighting coefficients about window--------------
            center_window = np.full((win_size[1], win_size[0]), cur_window[win_center[0], win_center[1]])
            gray_delta = cur_window - center_window

            gray_deviation = np.sqrt(np.sum(pow(gray_delta * reliable_mat, 2)) / np.count_nonzero(reliable_mat))
            if gray_deviation == 0:
                gray_deviation = sys.float_info.epsilon

            coe_gray = np.exp(-(pow(gray_delta, 2) * reliable_mat) / (neighbour_effect * gray_deviation))
            coe_dist = np.exp(-dist_window * reliable_mat)

            coe_window = coe_gray * coe_dist

            # --------------New intensity value of the center point in the window--------------
            new_gray = np.sum(coe_window * cur_window) / np.sum(coe_window)
            filtered_image.append(round(new_gray))

    filtered_image = np.array(filtered_image)
    return filtered_image


class FCM:
    def __init__(self, image, num_of_clusters, m, neighbor_effect, epsilon, max_iter, kernel):
        self.image = image
        self.num_of_clusters = num_of_clusters
        self.m = m
        self.neighbor_effect = neighbor_effect
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.kernel = kernel

        self.result = None
        self.centroid = None
        self.membership = None
        self.filtered_image = None
        self.histogram = None
        self.gray_num = None

    def init_membership(self):
        U = np.zeros((self.gray_num, self.num_of_clusters))
        grays = np.arange(self.gray_num)
        for clu_idx in range(self.num_of_clusters):
            bool_idx = grays % self.num_of_clusters == clu_idx
            U[bool_idx, clu_idx] = 1
            print(bool_idx)
        return U

    def update_membership(self):
        # Compute weights
        grays = np.arange(self.gray_num)
        c_mesh, idx_mesh = np.meshgrid(self.centroid, grays)
        power = -2. / (self.m - 1.)
        numerator = abs(idx_mesh - c_mesh) ** power
        denominator = np.sum(abs(idx_mesh - c_mesh) ** power, axis=1)
        return numerator / denominator[:, None]

    def update_centroid(self):
        # Compute centroid of clusters
        idx = np.arange(self.gray_num)
        idx_reshape = idx.reshape(len(idx), 1)
        numerator = np.sum(self.histogram * idx_reshape * pow(self.membership, self.m), axis=0)
        denominator = np.sum(self.histogram * pow(self.membership, self.m), axis=0)
        return numerator / denominator

    def get_filtered_image(self):
        # Create padding image
        pad_size_y = int(self.kernel[0] / 2)
        pad_size_x = int(self.kernel[1] / 2)
        padded_image = cv.copyMakeBorder(self.image, pad_size_y, pad_size_y, pad_size_x, pad_size_x,
                                         cv.BORDER_CONSTANT, value=0)  # zero padding
        filtered_image = sliding_window(padded_image, self.neighbor_effect, win_size=self.kernel)

        self.filtered_image = filtered_image.reshape(self.image.shape).astype(self.image.dtype)

    def calculate_histogram(self):
        hist = cv.calcHist([self.filtered_image], [0], None, [256], [0, 256])
        self.gray_num = len(hist)
        self.histogram = hist

    def defuzzify(self):
        return np.argmax(self.membership, axis=1)

    def generate_result(self):
        # Segment image based on max weights
        result = self.defuzzify()

        self.result = np.array(self.filtered_image, copy=True)
        for i in range(len(result)):
            self.result[self.result == i] = result[i]

        self.result = self.result.reshape(self.image.shape).astype('int')

        def random_color():
            return int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)

        height = self.image.shape[0]
        width = self.image.shape[1]

        colors = [random_color() for _ in range(self.num_of_clusters)]

        result = np.zeros((height, width, 3), np.uint8)
        for y in range(height):
            for x in range(width):
                result[y, x] = colors[self.result[y][x]]

        self.result = result

    def run(self):
        print("Getting filtered image...(This process can be time consuming.)")
        self.get_filtered_image()
        self.calculate_histogram()

        print("Iterative training...")
        self.membership = self.init_membership()
        itr = 0
        while True:
            self.centroid = self.update_centroid()
            pre_membership = np.copy(self.membership)
            self.membership = self.update_membership()
            err = np.sum(abs(self.membership - pre_membership))
            print("Iteration %d : cost = %f" % (itr, float(err)))

            if err < self.epsilon or itr > self.max_iter:
                break
            itr += 1
        self.generate_result()


def get_args():
    parser = argparse.ArgumentParser(description="Modified Fuzzy C-Means Algorithm")
    parser.add_argument('--input_path', type=str, default="./assets/images",
                        help='input path')
    parser.add_argument('--output_path', type=str, default="./assets/results",
                        help='output path')
    parser.add_argument('--num_of_clusters', type=int, default=4,
                        help="Number of cluster")
    parser.add_argument('--fuzziness', type=int, default=2,
                        help="fuzziness degree")
    parser.add_argument('--max_iteration', type=int, default=100,
                        help="max number of iterations.")
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help="threshold to check convergence.")

    parser.add_argument('--kernel', type=tuple, default=(5, 5),
                        help="Window size of MFCM algorithm")
    parser.add_argument('--neighbor_effect', type=float, default=3.,
                        help="Effect factor of the gray level, controls the influence extent of neighboring pixels.")

    args = parser.parse_args()
    return args


def main(args):
    input_file = const.input_file
    output_file = const.output_file
    image = cv.imread(input_file, cv.IMREAD_GRAYSCALE)

    # --------------Clustering--------------
    fcm = FCM(image, num_of_clusters=args.num_of_clusters, m=args.fuzziness, neighbor_effect=args.neighbor_effect,
              epsilon=args.epsilon, max_iter=args.max_iteration, kernel=args.kernel)
    fcm.run()
    result = fcm.result

    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image, cmap='gray')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(result, cmap='Accent', interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    main(get_args())
