import cv2 as cv
import const
import math
import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from itertools import cycle
from morphsnakes import recur_call


def inverse_gaussian_gradient(image, alpha=100.0, sigma=5.0):
    grad = ndi.gaussian_gradient_magnitude(image, sigma, mode='nearest')
    return 1.0 / np.sqrt(1.0 + alpha * grad)


class MorphGAC(object):
    """Morphological GAC based on the Geodesic Active Contours."""

    def __init__(self, image, smoothing=1, threshold=0.5, balloon=0, iteration=20):
        self.gamma = None
        self.smoothing = smoothing
        self.threshold = threshold
        self.balloon = balloon
        self.iteration = iteration

        self.image = inverse_gaussian_gradient(image, alpha=1000, sigma=2.0)
        self.update_mask()
        # The structure element for binary dilation and erosion.
        self.structure = np.ones((3,) * np.ndim(image))

        self.threshold_mask = None
        self.balloon_mask = None

    def set_gamma(self, ini):
        self.gamma = np.double(ini)
        self.gamma[ini > 0] = 1
        self.gamma[ini <= 0] = 0

    def set_balloon(self, ini):
        self.balloon = ini
        self.update_mask()

    def set_threshold(self, theta):
        self.threshold = theta
        self.update_mask()

    def update_mask(self):
        self.threshold_mask = self.image > self.threshold
        self.balloon_mask = self.image > self.threshold / np.abs(self.balloon)

    def step(self):
        gamma = self.gamma
        balloon = self.balloon
        image = self.image

        if gamma is None:
            raise ValueError("the initial level set function is not set ")

        # Balloon.
        ini = None
        if balloon > 0:
            ini = ndi.binary_dilation(gamma, self.structure)
        elif balloon < 0:
            ini = ndi.binary_erosion(gamma, self.structure)

        if balloon != 0:
            gamma[self.balloon_mask] = ini[self.balloon_mask]

        # Image attachment.
        ini = np.zeros_like(image)
        d_image = np.gradient(image)
        d_gamma = np.gradient(gamma)
        for el1, el2 in zip(d_image, d_gamma):
            ini = ini + el1 * el2

        gamma[ini > 0] = 1
        gamma[ini < 0] = 0

        # Smoothing.
        for _ in range(self.smoothing):
            gamma = recur_call(gamma)

        self.gamma = gamma

    def run(self, gamma):
        if self.gamma is None:
            self.gamma = gamma

        for _ in range(self.iteration):
            self.step()


# Chan-vese model
def chan_vese(phi, img, mu, nu, epsilon, step):
    delta = (epsilon / math.pi) / (epsilon * epsilon + phi * phi)
    hea_func = 0.5 * (1 + (2 / math.pi) * np.arctan(phi / epsilon))
    iy, ix = np.gradient(phi)
    s = np.sqrt(ix * ix + iy * iy)
    nx = ix / (s + 0.000001)
    ny = iy / (s + 0.000001)
    mxx, nxx = np.gradient(nx)
    nyy, myy = np.gradient(ny)
    cur = nxx + nyy
    length = nu * delta * cur
    lap = cv.Laplacian(phi, -1)
    penalty = mu * (lap - cur)
    s1 = hea_func * img
    s2 = (1 - hea_func) * img
    s3 = 1 - hea_func
    c1 = s1.sum() / hea_func.sum()
    c2 = s2.sum() / s3.sum()
    term = delta * ((img - c2) ** 2 - (img - c1) ** 2)
    phi = phi + step * (length + penalty + term)
    return phi


def main():
    origin = cv.imread("../assets/seg_test.jpg", cv.COLOR_BGRA2BGR)
    image = cv.cvtColor(origin, cv.COLOR_BGR2GRAY)
    image = np.array(image, dtype=np.float64)

    ini_gamma = np.ones((image.shape[0], image.shape[1]), image.dtype)
    ini_gamma[200:220, 200:220] = -1
    ini_gamma = -ini_gamma

    origin = cv.cvtColor(origin, cv.COLOR_BGR2RGB)
    plt.figure(1)
    plt.imshow(origin)
    plt.xticks([])
    plt.yticks([])  # to hide tick values on X and Y axis
    plt.contour(ini_gamma, [1])
    plt.draw()
    plt.show(block=False)

    mu = 1
    nu = 0.003 * 255 * 255
    num = 5
    epsilon = 0.5
    step = 0.05

    gamma = ini_gamma
    for _ in range(1, num):
        gamma = chan_vese(gamma, image, mu, nu, epsilon, step)

    plt.imshow(origin), plt.xticks([]), plt.yticks([])
    plt.contour(gamma, [0.5], colors='r')
    plt.draw(), plt.show(), plt.pause(0.01)
