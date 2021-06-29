import argparse
import const
import cv2 as cv
import numpy as np

from itertools import cycle
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from dummy import Dummy, merge
from assess import *


class Recur:
    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)


P = [np.eye(3),
     np.array([[0, 1, 0]] * 3),
     np.flipud(np.eye(3)),
     np.rot90([[0, 1, 0]] * 3)]

u_x = np.zeros(0)


def sup_inf(gamma):
    global u_x
    if np.ndim(gamma) == 2:
        p = P
    else:
        raise ValueError("invalid number of dimensions, should be 2)")

    if gamma.shape != u_x.shape[0:]:
        u_x = np.zeros((len(p),) + gamma.shape)

    for u_i, p_i in zip(u_x, p):
        u_i[:] = ndi.binary_erosion(gamma, p_i)

    return np.amax(u_x, axis=0)


def inf_sup(gamma):
    global u_x
    if np.ndim(gamma) == 2:
        p = P
    else:
        raise ValueError("invalid number of dimensions, should be 2")

    if gamma.shape != u_x.shape[0:]:
        u_x = np.zeros((len(p),) + gamma.shape)

    for u_i, p_i in zip(u_x, p):
        u_i[:] = ndi.binary_dilation(gamma, p_i)

    return np.amin(u_x, axis=0)


def si_o_is(gamma):
    return sup_inf(inf_sup(gamma))


def is_o_si(gamma):
    return inf_sup(sup_inf(gamma))


recur_call = Recur([si_o_is, is_o_si])


def inverse_gaussian_gradient(image, alpha=100.0, sigma=5.0):
    grad = ndi.gaussian_gradient_magnitude(image, sigma, mode='nearest')
    return 1.0 / np.sqrt(1.0 + alpha * grad)


def circle_level_set(image_shape, center=None, radius=None):
    if center is None:
        center = tuple(i // 2 for i in image_shape)

    if radius is None:
        radius = min(image_shape) * 3.0 / 8.0

    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum(grid ** 2, 0))
    level_set = np.int8(phi > 0)
    return level_set


def checkerboard_level_set(image_shape, square_size=5):
    grid = np.ogrid[[slice(i) for i in image_shape]]
    grid = [(grid_i // square_size) & 1 for grid_i in grid]

    grid = np.array(grid, dtype=object)
    checkerboard = np.bitwise_xor.reduce(grid, axis=0)
    level_set = np.int8(checkerboard)

    return level_set


def dummy_level_set(image):
    dummy = Dummy(image)
    dummy.run()

    return dummy.result


def dummy_image(image):
    dummy = Dummy(image)
    dummy.run()

    return dummy.image


class MorphACWE(object):
    """Morphological ACWE based on the Chan-Vese energy functional."""

    def __init__(self, image, smoothing=1, lambda1=1, lambda2=1, iteration=20):
        self.image = image
        self.smoothing = smoothing
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.iteration = iteration

        self.gamma = None

    def set_gamma(self, ini):
        self.gamma = np.double(ini)
        self.gamma[ini <= 4] = 0
        self.gamma[ini > 4] = 1

    def step(self):
        """Perform a single step of the morphological Chan-Vese evolution."""

        gamma = self.gamma
        if gamma is None:
            raise ValueError("the initial level set function is not set ")

        image = self.image

        # determine c0 and c1.
        outside = (gamma <= 0)
        inside = (gamma > 0)
        c0 = image[outside].sum() / float(outside.sum())
        c1 = image[inside].sum() / float(inside.sum())

        # attraction.
        grad = np.array(np.gradient(gamma))
        grad = np.abs(grad).sum(0)
        # gradient direction.
        ini = grad * (self.lambda1 * (image - c1) ** 2 -
                      self.lambda2 * (image - c0) ** 2)

        # <0 inside.
        gamma[ini < 0] = 1
        # >0 outside.
        gamma[ini > 0] = 0

        # Smoothing.
        for _ in range(self.smoothing):
            gamma = recur_call(gamma)

        self.gamma = gamma

    def run(self, gamma):
        if self.gamma is None:
            self.gamma = gamma

        for _ in range(self.iteration):
            self.step()


def get_args():
    parser = argparse.ArgumentParser(description='Morphological ACWE based on the Chan-Vese energy functional')
    parser.add_argument('--input_path', type=str, default="./assets/images",
                        help='input path')
    parser.add_argument('--output_path', type=str, default="./assets/results",
                        help='output path')
    parser.add_argument('--smoothing', type=int, default=2,
                        help='a constant to control the details in gamma, values(1 - 4)')
    parser.add_argument('--iteration', type=int, default=5,
                        help='max iteration for evolution')

    parser.add_argument('--kernel', type=tuple, default=(5, 5),
                        help='a tuple for the Gaussian Filter')

    args = parser.parse_args()

    return args


def main(args):
    input_file = const.input_file()
    output_file = const.output_file()
    image = cv.imread(input_file, cv.IMREAD_GRAYSCALE)

    # image = cv.GaussianBlur(image, const.kernel(), 0)

    # ms = MorphACWE(image, smoothing=args.smoothing, iteration=args.iteration)
    ms = MorphACWE(dummy_image(image), smoothing=args.smoothing, lambda2=2, iteration=args.iteration)

    # gamma = circle_level_set(image.shape)
    # gamma = checkerboard_level_set(image.shape)
    gamma = dummy_level_set(image)
    thr_res = gamma

    assess(const.get_target(), gamma)

    ms.run(gamma=gamma)
    result = ms.gamma

    fig = plt.figure()
    fig.clf()
    ax0 = fig.add_subplot(1, 3, 1)
    # ax0.imshow(result, cmap='Paired', interpolation='nearest')
    ax0.imshow(image, cmap='gray')
    ax0.set_axis_off()

    ms = MorphACWE(dummy_image(image), smoothing=args.smoothing, lambda2=5, iteration=args.iteration)

    ms.run(gamma=gamma)
    # result = ms.gamma

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.imshow(image, cmap='gray')
    # ax1.contour(result, [0.5], colors='r')
    ax1.imshow(result, cmap='Paired', interpolation='nearest')
    ax1.set_axis_off()

    result = merge(result, thr_res)

    # result = cv.morphologyEx(result, cv.MORPH_OPEN, const.kernel())
    # result = cv.morphologyEx(result, cv.MORPH_CLOSE, const.kernel())
    # result = cv.dilate(result, const.kernel(), iterations=5)

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.imshow(result, cmap='Paired', interpolation='nearest')
    ax2.set_axis_off()
    plt.show()

    assess(const.get_target(), result)


if __name__ == '__main__':
    main(get_args())
