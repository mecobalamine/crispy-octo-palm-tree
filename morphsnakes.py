import argparse
import const
import cv2 as cv
import numpy as np

from itertools import cycle
from scipy import ndimage as ndi
from matplotlib import pyplot as plt


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


def checkerboard_level_set(image_shape, square_size=5):
    grid = np.ogrid[[slice(i) for i in image_shape]]
    grid = [(grid_i // square_size) & 1 for grid_i in grid]

    grid = np.array(grid, dtype=object)
    checkerboard = np.bitwise_xor.reduce(grid, axis=0)
    level_set = np.int8(checkerboard)

    return level_set


class MorphACWE:
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
        self.gamma[ini <= 0] = 0
        self.gamma[ini > 0] = 1

    def step(self):
        """Perform a single step of the morphological Chan-Vese evolution."""
        # Assign attributes to local variables for convenience.
        gamma = self.gamma
        if gamma is None:
            raise ValueError("the initial level set function is not set ")

        image = self.image

        # Determine c0 and c1.
        outside = (gamma <= 0)
        inside = (gamma > 0)
        c0 = image[outside].sum() / float(outside.sum())
        c1 = image[inside].sum() / float(inside.sum())

        # Image attachment.
        delta = np.array(np.gradient(gamma))
        delta = np.abs(delta).sum(0)
        ini = delta * (self.lambda1 * (image - c1) ** 2 -
                       self.lambda2 * (image - c0) ** 2)

        gamma[ini < 0] = 1
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
    input_file = const.input_file
    output_file = const.output_file
    image = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    image = cv.GaussianBlur(image, args.kernel, 0)

    ms = MorphACWE(image, smoothing=args.smoothing, iteration=args.iteration)
    ms.run(gamma=checkerboard_level_set(image.shape))
    result = ms.gamma

    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image, cmap='gray')
    ax1.contour(result, [0.5], colors='r')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(result, cmap='Paired', interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    main(get_args())
