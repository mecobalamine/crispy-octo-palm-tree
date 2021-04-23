import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

Image = cv2.imread("./assets/seg_test.jpg", 1)  # 读入原图
image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
img = np.array(image, dtype=np.float64)  # 读入到np的array中，并转化浮点类型

# 初始水平集函数
IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype)
IniLSF[200:220, 200:220] = -1
IniLSF = -IniLSF

# 画初始轮廓
Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
plt.figure(1), plt.imshow(Image), plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.contour(IniLSF, [0.5])  # 画LSF=0处的等高线
plt.draw(), plt.show(block=False)


# CV函数
def CV(phi, img, mu, nu, epsilon, step):
    delta = (epsilon / math.pi) / (epsilon * epsilon + phi * phi)
    hea_func = 0.5 * (1 + (2 / math.pi) * np.arctan(phi / epsilon))
    Iy, Ix = np.gradient(phi)
    s = np.sqrt(Ix * Ix + Iy * Iy)
    Nx = Ix / (s + 0.000001)
    Ny = Iy / (s + 0.000001)
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)
    cur = Nxx + Nyy
    length = nu * delta * cur

    Lap = cv2.Laplacian(phi, -1)
    Penalty = mu * (Lap - cur)

    s1 = hea_func * img
    s2 = (1 - hea_func) * img
    s3 = 1 - hea_func
    c1 = s1.sum() / hea_func.sum()
    c2 = s2.sum() / s3.sum()
    CVterm = delta * ((img - c2) ** 2 - (img - c1) ** 2)

    phi = phi + step * (length + Penalty + CVterm)
    return phi


# 模型参数
mu = 1
nu = 0.003 * 255 * 255
num = 5
epsilon = 0.5
step = 0.05
LSF = IniLSF
for _ in range(1, num):
    LSF = CV(LSF, img, mu, nu, epsilon, step)  # 迭代
plt.imshow(Image), plt.xticks([]), plt.yticks([])
plt.contour(LSF, [0.5], colors='r')
plt.draw(), plt.show(), plt.pause(0.01)
