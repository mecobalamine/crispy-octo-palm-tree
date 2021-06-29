import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import math
import functools
import copy


class KeyPoint:
    x = -1
    y = -1
    octave = -1
    layer = -1
    xi = 0
    size = 0
    response = 0
    angle = 0


def keypoint_sort_size(kpt1, kpt2):
    if kpt1.size > kpt2.size:
        return 1
    if kpt1.size < kpt2.size:
        return -1
    return 0


def keypoint_sort_resbonse(kpt1, kpt2):
    if kpt1.response > kpt2.response:
        return -1
    if kpt1.response < kpt2.response:
        return 1
    return 0


def keypoint_less(kpt1, kpt2):
    if kpt1.x > kpt2.x:
        return 1
    if kpt1.x < kpt2.x:
        return -1
    if kpt1.y > kpt2.y:
        return 1
    if kpt1.y < kpt2.y:
        return -1
    if kpt1.size > kpt2.size:
        return 1
    if kpt1.size < kpt2.size:
        return -1
    if kpt1.angle > kpt2.angle:
        return 1
    if kpt1.angle < kpt2.angle:
        return -1
    if kpt1.response > kpt2.response:
        return 1
    if kpt1.response < kpt2.response:
        return -1
    if kpt1.octave > kpt2.octave:
        return 1
    if kpt1.octave < kpt2.octave:
        return -1
    return 0


class SIFT:
    SIFT_DESCR_WIDTH = 4
    SIFT_DESCR_HIST_BINS = 8
    SIFT_INIT_SIGMA = 0.5
    SIFT_IMG_BORDER = 5
    SIFT_MAX_INTERP_STEPS = 5
    SIFT_ORI_HIST_BINS = 36
    SIFT_ORI_SIG_FCTR = 1.5
    SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR
    SIFT_ORI_PEAK_RATIO = 0.8
    SIFT_DESCR_SCL_FCTR = 3.
    SIFT_DESCR_MAG_THR = 0.2
    SIFT_INT_DESCR_FCTR = 512.
    SIFT_FIXPT_SCALE = 1.
    KPT_ANGLE = 0

    def __init__(self, nfeatures=30,
                 nOctaveLayers=3,
                 contrastThreshold=0.03,
                 edgeThreshold=10,
                 sigma=1.6
                 ):
        self._nfeatures = nfeatures
        self._nOctaveLayers = nOctaveLayers
        self._contrastThreshold = contrastThreshold
        self._edgeThreshold = edgeThreshold
        self._sigma = sigma

    def create_initial_image(self, image_in):
        '''
        创建第一组图片，因为放大为原始图两倍，sigma也要乘2
        因为作者假设原始图片经过了至少高斯滤波sigma = 0.5，doubled image sigma = 1.0所以要做差
        opencv里面直接resize而不是插值
        '''
        sig_diff = np.sqrt(np.max([(self._sigma) ** 2 - (self.SIFT_INIT_SIGMA * 2) ** 2, 0.01]))
        resized_img = cv2.resize(image_in, (image_in.shape[1] * 2, image_in.shape[0] * 2), interpolation=cv2.INTER_AREA)
        resized_img = cv2.GaussianBlur(resized_img, (0, 0), sig_diff)
        return resized_img

    def build_gaussian_pyramid(self, image_in, nOctaves):
        """
        第一层图片的初始尺度为sig = 1.6,后面每一张尺度相差k = 2**(1/3),第三张为2**1即 ,最后一张sigma = 1.6*2**(5/3)
        第二层尺度初始为2**1.6，以此类推，

        """
        image = image_in.copy()
        # must produce s + 3 images in the stack of blurred images for each octave
        # s+3 = 6,高斯差分为5层，求导为4层，正好
        sig = np.zeros(self._nOctaveLayers + 3, dtype=np.float32)
        pyr = {}
        k = 2 ** (1 / self._nOctaveLayers)
        sig[0] = self._sigma
        for i in range(1, sig.shape[0]):
            sig_prev = k ** (i - 1) * self._sigma
            sig_toal = sig_prev * k
            sig[i] = np.sqrt(sig_toal ** 2 - sig_prev ** 2)

        for o in range(nOctaves):
            one_octave = np.zeros((image.shape[0] // (2 ** o), image.shape[1] // (2 ** o), (self._nOctaveLayers + 3)),
                                  dtype=np.float32)
            for l in range(self._nOctaveLayers + 3):
                if o == 0 and l == 0:
                    one_octave[:, :, 0] = image
                elif l == 0:
                    src_layer = self._nOctaveLayers
                    src = pyr[o - 1][:, :, src_layer].copy()
                    one_octave[:, :, l] = cv2.resize(src, (one_octave.shape[1], one_octave.shape[0]),
                                                     interpolation=cv2.INTER_AREA)

                else:
                    src_layer = l - 1
                    one_octave[:, :, l] = cv2.GaussianBlur(one_octave[:, :, src_layer], (0, 0),
                                                           sig[l])  # sig[l]相当于sig_diff
            pyr[o] = one_octave
        return pyr

    def build_dog_pyramid(self, gpyr):
        """
        每一组图片之间求差
        """
        dogpyr = {}
        for key, value in gpyr.items():
            one_dog = np.zeros((gpyr[key].shape[0], gpyr[key].shape[1], gpyr[key].shape[2] - 1), dtype=np.float32)
            for i in range(gpyr[key].shape[2] - 1):
                one_dog[:, :, i] = gpyr[key][:, :, i + 1] - gpyr[key][:, :, i]
            dogpyr[key] = one_dog
        return dogpyr

    def adjust_local_extrema(self, dogpyr, kpt, octave, layers, l, r, c):
        img_scale = 1 / 255 * self.SIFT_FIXPT_SCALE  # 求梯度的时候从0~255归一化到0~1
        deriv_scale = img_scale * 0.5
        second_deriv_scale = img_scale
        cross_deriv_scale = img_scale * 0.25
        xi = 0;
        xr = 0;
        xc = 0;
        # 迭代5次
        iter_cnt = 0
        rows = layers[:, :, l].shape[0]
        cols = layers[:, :, l].shape[1]
        for i in range(self.SIFT_MAX_INTERP_STEPS):
            iter_cnt += 1  # 记录迭代次数
            # 一阶导数
            dD = np.zeros(3, dtype=np.float32)  # dx,dy,d_sigma
            dD[0] = (layers[r, c + 1, l] - layers[r, c - 1, l]) * deriv_scale
            dD[1] = (layers[r + 1, c, l] - layers[r - 1, c, l]) * deriv_scale
            dD[2] = (layers[r, c, l + 1] - layers[r, c, l - 1]) * deriv_scale
            # 二阶导数
            v2 = layers[r, c, l] * 2
            dxx = (layers[r, c + 1, l] + layers[r, c - 1, l] - v2) * second_deriv_scale
            dyy = (layers[r + 1, c, l] + layers[r - 1, c, l] - v2) * second_deriv_scale
            dss = (layers[r, c, l + 1] + layers[r, c, l - 1] - v2) * second_deriv_scale
            dxy = (layers[r + 1, c + 1, l] - layers[r + 1, c - 1, l] - layers[r - 1, c + 1, l] + layers[
                r - 1, c - 1, l]) * cross_deriv_scale
            dxs = (layers[r, c + 1, l + 1] - layers[r, c - 1, l + 1] - layers[r, c + 1, l - 1] + layers[
                r, c - 1, l - 1]) * cross_deriv_scale
            dys = (layers[r + 1, c, l + 1] - layers[r - 1, c, l + 1] - layers[r + 1, c, l - 1] + layers[
                r - 1, c, l - 1]) * cross_deriv_scale

            H = np.array([[dxx, dxy, dxs],
                          [dxy, dyy, dys],
                          [dxs, dys, dss]], dtype=np.float32)
            X = np.dot(np.linalg.inv(H), dD)
            xi = -X[2]
            xr = -X[1]
            xc = -X[0]

            if abs(xi) < 0.5 and abs(xr) < 0.5 and abs(xc) < 0.5:
                break
            if abs(xi) > 100 or abs(xr) > 100 or abs(xc) > 100:
                return r, c, False
            # 如果偏移超过一个像素就要调整了
            c = int(c + round(xc))
            r = int(r + round(xr))
            l = int(l + round(xi))

            if l < 1 or l > self._nOctaveLayers \
                    or c < self.SIFT_IMG_BORDER or c > cols - self.SIFT_IMG_BORDER \
                    or r < self.SIFT_IMG_BORDER or r > rows - self.SIFT_IMG_BORDER:
                return r, c, False

        if iter_cnt >= self.SIFT_MAX_INTERP_STEPS:
            return r, c, False
        # 因为r，c，sigma有更新，重新算一次
        '''D_x_hat > _contrastThreshold'''
        dD = np.zeros(3, dtype=np.float32)  # dx,dy,d_sigma
        dD[0] = (layers[r, c + 1, l] - layers[r, c - 1, l]) * deriv_scale
        dD[1] = (layers[r + 1, c, l] - layers[r - 1, c, l]) * deriv_scale
        dD[2] = (layers[r, c, l + 1] - layers[r, c, l - 1]) * deriv_scale
        t = dD[0] * xc + dD[1] * xr + dD[2] * xi
        D_x_hat = layers[r, c, l] * img_scale + t * 0.5
        if abs(D_x_hat) * self._nOctaveLayers < self._contrastThreshold:
            return r, c, False

        '''Eliminating edge responses'''
        v2 = layers[r, c, l] * 2
        dxx = (layers[r, c + 1, l] + layers[r, c - 1, l] - v2) * second_deriv_scale
        dyy = (layers[r + 1, c, l] + layers[r - 1, c, l] - v2) * second_deriv_scale
        dxy = (layers[r + 1, c + 1, l] - layers[r + 1, c - 1, l] - layers[r - 1, c + 1, l] + layers[
            r - 1, c - 1, l]) * cross_deriv_scale
        tr = dxx + dyy
        det = dxx * dyy - dxy * dxy
        if det <= 0 or tr * tr * self._edgeThreshold >= (self._edgeThreshold + 1) * (self._edgeThreshold + 1) * det:
            return r, c, False
        kpt.x = (c + xc) * 2 ** (octave)  # 相对第一层的位置，后面有缩放
        kpt.y = (r + xr) * 2 ** (octave)
        kpt.octave = octave
        kpt.layer = l
        kpt.xi = xi
        kpt.size = self._sigma * 2 ** ((l + xi) / self._nOctaveLayers) * 2 ** (octave) * 2  # 这里乘2大概是半径*2
        kpt.response = abs(D_x_hat)
        return r, c, True

    def calc_orientation_hist(self, img_gaussion, r1, c1, radius, sigma, hist, n):
        expf_scale = 1 / (2 * sigma * sigma)
        data_len = (radius * 2 + 1) * (radius * 2 + 1)
        # block = np.zeros(data_len,dtype = np.float32)#在这个范围内确定方向
        Mag = np.zeros(data_len, dtype=np.float32)  # 梯度大小
        Ori = np.zeros(data_len, dtype=np.float32)  # 方向
        X = np.zeros(data_len, dtype=np.float32)  # x 梯度
        Y = np.zeros(data_len, dtype=np.float32)  # y 梯度
        W = np.zeros(data_len, dtype=np.float32)  # y 高斯分布权重
        temphist = np.zeros(n + 4, dtype=np.float32)  # 因为需要平滑
        k = 0
        for i in range(-radius, radius + 1):
            y = r1 + i
            if y < 0 or y >= img_gaussion.shape[0] - 1:
                continue
            for j in range(-radius, radius + 1):
                x = c1 + j
                if x < 0 or x >= img_gaussion.shape[1] - 1:
                    continue
                dx = img_gaussion[y, x + 1] - img_gaussion[y, x - 1]
                dy = img_gaussion[y - 1, x] - img_gaussion[y + 1, x]
                X[k] = dx
                Y[k] = dy
                W[k] = (i * i + j * j) * expf_scale
                k += 1
        for i in range(k):
            Ori[i] = (math.atan2(Y[i], X[i]) / 3.1415 * 180 + 360) % 360

            W[i] = np.exp(W[i])
            Mag[i] = np.sqrt(Y[i] ** 2 + X[i] ** 2)

        for i in range(k):
            bin = int(round(n / 360 * Ori[i]))
            if bin >= n:
                bin -= n
            if bin < 0:
                bin += n
            try:
                temphist[bin + 2] += (W[i] * Mag[i])
            except:
                print(temphist[bin + 2], W[i] * Mag[i])

        temphist[1] = temphist[n + 1]
        temphist[0] = temphist[n]
        temphist[n + 2] = temphist[2]
        temphist[n + 3] = temphist[3]

        for i in range(n):
            hist[i] = (temphist[i] + temphist[i + 4]) * 1 / 16 \
                      + (temphist[i + 1] + temphist[i + 3]) * 4 / 16 \
                      + (temphist[i + 2]) * 6 / 16
        return np.max(hist)

    def find_scale_space_extrema(self, gpyr, dogpyr):
        # Detects features at extrema in DoG scale space.  Bad features are discarded
        # based on contrast and ratio of principal curvatures.
        """
        寻找最值点，调整极值点位置，根据contrast筛选，特征点方向
        """
        # nOctaves = len(dogpyr)
        keypoints = []
        threshold = int(0.5 * self._contrastThreshold / self._nOctaveLayers * 255 * self.SIFT_FIXPT_SCALE)
        n = self.SIFT_ORI_HIST_BINS
        hist = np.zeros(n, dtype=np.float32)

        cnt = 0
        for octave, layers in dogpyr.items():
            for i in range(1, self._nOctaveLayers + 1):
                # prev_ind = i-1
                # next_ind = i+1
                rows = layers[:, :, i].shape[0]
                cols = layers[:, :, i].shape[1]

                for r in range(self.SIFT_IMG_BORDER, rows - self.SIFT_IMG_BORDER):
                    for c in range(self.SIFT_IMG_BORDER, cols - self.SIFT_IMG_BORDER):
                        neighbor = layers[r - 1:r + 2, c - 1:c + 2, i - 1:i + 2]
                        cur_val = layers[r, c, i]
                        # 最值点
                        if abs(cur_val) > threshold and \
                                ((cur_val > threshold and cur_val == np.max(neighbor)) \
                                 or (cur_val < threshold and cur_val == np.min(neighbor))):
                            kpt = KeyPoint()
                            r1, c1, isKpt = self.adjust_local_extrema(dogpyr, kpt, octave, layers, i, r, c, )
                            if isKpt == False:
                                continue
                            # print(kpt.size,r1,r1)
                            scl_octv = kpt.size * 0.5 / (1 << octave)
                            img_gaussion = gpyr[octave][:, :, i]
                            radius = int(self.SIFT_ORI_RADIUS * scl_octv)
                            sigma = scl_octv * self.SIFT_ORI_SIG_FCTR
                            omax = self.calc_orientation_hist(img_gaussion, r1, c1, radius, sigma, hist, n)
                            # print(omax,hist)
                            mag_thr = omax * self.SIFT_ORI_PEAK_RATIO  # 作者说只要是大于80%的方向都要取一个kpt,大概15%的点会对应多个特征点
                            for j in range(n):
                                l = (j + n - 1) % n
                                r2 = (j + n + 1) % n
                                if hist[j] > hist[l] and hist[j] > hist[r2] and hist[j] > mag_thr:
                                    bin = j + 0.5 * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[j] + hist[r2])
                                    bin = (bin + n) % n
                                    # kpt.angle = 360 - ((360/n) * bin)
                                    kpt.angle = (360 / n) * bin  # self.KPT_ANGLE#((360/n) * bin)
                                    if abs(kpt.angle - 360) < 0.00001:
                                        kpt.angle = 0
                                    keypoints.append(kpt)

        # return np.asarray(keypoints)
        return keypoints

    def remove_duplicated(self, keypoints):
        keypoints = sorted(keypoints, key=functools.cmp_to_key(keypoint_less))
        n = len(keypoints)
        kpidx = np.arange(n)
        mask = np.ones(n, dtype=np.uint8)
        j = 0
        for i in range(1, n):
            kp1 = keypoints[kpidx[i]]
            kp2 = keypoints[kpidx[j]]
            if kp1.x - kp2.x < 0.0001 and kp1.y - kp2.y < 0.0001 \
                    and kp1.size - kp2.size < 0.0001 and kp1.angle - kp2.angle < 0.0001:
                mask[kpidx[i]] = -1
            else:
                j = i
        j = 0
        for i in range(n):
            if mask[i] >= 0:
                if i != j:
                    keypoints[j] = keypoints[i]
                j += 1
        keypoints = keypoints[:j]
        return keypoints

    def retain_best_keypoint(self, keypoints):
        if self._nfeatures < 1 or len(keypoints) <= self._nfeatures:
            return keypoints
        keypoints = sorted(keypoints, key=functools.cmp_to_key(keypoint_sort_resbonse))
        keypoints = keypoints[:self._nfeatures]
        return keypoints

    def calc_sift_descriptor(self, img, x, y, ori, scl, d, n, scale):
        x = int(round(x))
        y = int(round(y))
        # print(ori,"rotate---------------------------->")
        cos_t = np.cos(ori / 180 * 3.141592654)
        sin_t = np.sin(ori / 180 * 3.141592654)

        # j = int(100*np.cos(ori/180*3.141592654))#x
        # i = int(100*np.sin(ori/180*3.141592654))#y
        # print("ex",(math.atan2(i, j)/3.1415*180 + 360)%360,j,i)
        # r_rot1 = i * cos_t - j * sin_t#y
        # c_rot1 = i * sin_t + j * cos_t#x
        # print(c_rot1,r_rot1)
        # print("now",(math.atan2(r_rot1, c_rot1)/3.1415*180 + 360)%360)

        bins_per_rad = n / 360
        exp_scale = -1 / (d * d * 0.5)  # 算权重用
        hist_width = self.SIFT_DESCR_SCL_FCTR * scl
        radius = round(hist_width * 1.414213562373 * (d + 1) * 0.5)  # 在这个半径旋转
        radius = int(np.min([radius, np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)]))
        cos_t /= hist_width  # 缩放旋转后的pixel到指定范围内
        sin_t /= hist_width
        data_len = (radius * 2 + 1) * (radius * 2 + 1)
        Mag = np.zeros(data_len, dtype=np.float32)  # 梯度大小
        Ori = np.zeros(data_len, dtype=np.float32)  # 方向
        X = np.zeros(data_len, dtype=np.float32)  # x 梯度
        Y = np.zeros(data_len, dtype=np.float32)  # y 梯度
        W = np.zeros(data_len, dtype=np.float32)  # y 高斯分布权重
        RBin = np.zeros(data_len, dtype=np.float32)  # 旋转后的点
        CBin = np.zeros(data_len, dtype=np.float32)
        histlen = (d + 2) * (d + 2) * (n + 2)
        hist = np.zeros(histlen, dtype=np.float32)
        dst = np.zeros(d * d * n, dtype=np.float32)
        # hist =  np.zeros(((d+2),(d+2),(n+2)),dtype = np.float32)#多出来的两行插值用
        rows = img.shape[0]
        cols = img.shape[1]

        k = 0
        box = []
        # 旋转后的左边在半径为d/2范围内
        for i in range(-radius, radius + 1):  # y,row
            for j in range(-radius, radius + 1):  # x,col
                # print("旋转前",(math.atan2(i, j)/3.1415*180 + 360)%360,j,i)
                r_rot = i * cos_t - j * sin_t
                c_rot = i * sin_t + j * cos_t
                # print("旋转后",(math.atan2(r_rot, c_rot)/3.1415*180 + 360)%360,c_rot,r_rot)
                rbin = r_rot + d / 2 - 0.5
                cbin = c_rot + d / 2 - 0.5
                r = y + i
                c = x + j
                if radius == abs(i) and radius == abs(j):
                    box.append([round((cbin + x) / scale), round((r_rot + y) / scale)])
                    # box.append(r_rot + y)/scale
                if -1 < rbin < d - 0.1 and -1 < cbin < d - 0.1 and \
                        0 < r < rows - 1 and 0 < c < cols - 1:
                    dx = img[r, c + 1] - img[r, c - 1]
                    dy = img[r - 1, c] - img[r + 1, c]
                    X[k] = dx
                    Y[k] = dy
                    RBin[k] = rbin
                    CBin[k] = cbin
                    W[k] = (c_rot * c_rot + r_rot * r_rot) * exp_scale  # 距离中心的距离
                    k += 1
        for i in range(k):
            Ori[i] = ((math.atan2(Y[i], X[i]) / 3.1415 * 180 + ori) + 360) % 360
            Mag[i] = np.sqrt(Y[i] ** 2 + X[i] ** 2)
            W[i] = np.exp(W[i])

        for i in range(k):
            rbin = RBin[i]
            cbin = CBin[i]
            obin = (Ori[i]) * bins_per_rad  # 旋转坐标轴后方向也变化
            mag = Mag[i] * W[i]  # z理解为hist像素的大小，对该值进行插值

            r0 = int(rbin)
            c0 = int(cbin)
            o0 = int(obin)

            rbin -= r0
            cbin -= c0
            obin -= o0

            o0 = (o0 + n) % n

            # 这部分是三线性插值，将mag按照和周围8个hist点的距离分配
            v_r1 = mag * rbin;
            v_r0 = mag - v_r1
            v_rc11 = v_r1 * cbin;
            v_rc10 = v_r1 - v_rc11
            v_rc01 = v_r0 * cbin;
            v_rc00 = v_r0 - v_rc01
            v_rco111 = v_rc11 * obin;
            v_rco110 = v_rc11 - v_rco111
            v_rco101 = v_rc10 * obin;
            v_rco100 = v_rc10 - v_rco101
            v_rco011 = v_rc01 * obin;
            v_rco010 = v_rc01 - v_rco011
            v_rco001 = v_rc00 * obin;
            v_rco000 = v_rc00 - v_rco001

            idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0
            hist[idx] += v_rco000
            hist[idx + 1] += v_rco001
            hist[idx + (n + 2)] += v_rco010
            hist[idx + (n + 3)] += v_rco011
            hist[idx + (d + 2) * (n + 2)] += v_rco100
            hist[idx + (d + 2) * (n + 2) + 1] += v_rco101
            hist[idx + (d + 3) * (n + 2)] += v_rco110
            hist[idx + (d + 3) * (n + 2) + 1] += v_rco111

        # 角度是循环的，所以要给最终描述子要加上循环部分的hist
        for i in range(d):
            for j in range(d):
                idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2)
                hist[idx] += hist[idx + n]
                hist[idx + 1] += hist[idx + n + 1]
                for k1 in range(n):
                    dst[(i * d + j) * n + k1] = hist[idx + k1]

        thr = np.sqrt(np.sum(dst * dst)) * self.SIFT_DESCR_MAG_THR  # 0.2
        dst = np.clip(dst, a_min=0, a_max=thr)
        nrm = 1 / np.sqrt(np.sum(dst * dst))
        # dst = np.clip(dst*nrm,a_min = 0, a_max = 255)
        dst = dst * nrm
        return dst, box

    def calc_descriptors(self, gpyr, keypoints):
        d = self.SIFT_DESCR_WIDTH
        n = self.SIFT_DESCR_HIST_BINS
        descriptors = []
        boxes = []
        for i in range(len(keypoints)):
            kpt = copy.copy(keypoints[i])
            layer = int(kpt.layer)
            octave = int(kpt.octave)
            scale = 1 / (2 ** octave)
            x = kpt.x * scale
            y = kpt.y * scale
            size = kpt.size * scale
            img = gpyr[octave + 1][:, :, layer]
            # angle = 360 - kpt.angle
            angle = kpt.angle
            if abs(angle - 360) < 0.00001:
                angle = 0
            # 这里size*0.5因为之前adjustLocalExtrema size*2
            #            print(img.shape, x , y, angle,size*0.5,d, n,scale)

            hist, box = self.calc_sift_descriptor(img, x, y, angle, size * 0.5, d, n, scale)
            boxes.append(box)
            des = {'x': kpt.x,
                   'y': kpt.y,
                   'hist': hist}
            descriptors.append(des)
        return descriptors, boxes

    def run_sift_feaures(self, image):
        # 输入图像0~255
        # plt.imshow(image,"gray"),plt.title("original img"),plt.show()
        initial_image = self.create_initial_image(image)
        # plt.imshow(initial_image,"gray"),plt.title("image -1 octave double size"),plt.show()
        nOctaves = int(np.min([np.log(initial_image.shape[0]), np.log(initial_image.shape[1])] / np.log(2) - 2) + 1)
        gpyr = self.build_gaussian_pyramid(initial_image, nOctaves)
        dogpyr = self.build_dog_pyramid(gpyr)
        # for key, value in dogpyr.items():
        #     for i in range(value.shape[2]):
        #          plt.imshow(dogpyr[key][:,:,i],"gray"),plt.title("gpyr"),plt.show()
        keypoints = self.find_scale_space_extrema(gpyr, dogpyr)
        print("去重前", len(keypoints))
        keypoints = self.remove_duplicated(keypoints)
        print("去重后", len(keypoints))
        keypoints = self.retain_best_keypoint(keypoints)

        for i in range(len(keypoints)):
            # 很苦恼在这一步如果按照注释的做会数据异常！
            scale = 0.5  # 第一层放大了两倍
            # keypoints[i].x = keypoints[i].x * scale
            # keypoints[i].y = keypoints[i].y * scale
            # keypoints[i].size = keypoints[i].size * scale
            temp = copy.copy(keypoints[i])
            temp.x = keypoints[i].x * scale
            temp.y = keypoints[i].y * scale
            temp.size = keypoints[i].size * scale
            # print(temp.x,temp.y)
            keypoints[i] = temp
        descriptors, boxes = self.calc_descriptors(gpyr, keypoints)

        return keypoints, descriptors, boxes


def main():
    input = 'lena1.jpg'
    img = cv2.imread(input, 0)
    img1 = cv2.imread(input)
    start = time.time()
    s = SIFT(nfeatures=0,
             edgeThreshold=10,
             contrastThreshold=0.04)
    s.KPT_ANGLE = 0
    keypoints, descriptors, boxes = s.run_sift_feaures(img)

    for i in range(len(keypoints)):
        cv2.circle(img1, (int(keypoints[i].x), int(keypoints[i].y)), int(keypoints[i].size), (0, 255, 0), 1)
        ptStart = (int(keypoints[i].x), int(keypoints[i].y))
        ptEnd = (int(keypoints[i].x) + int(10 * np.cos(keypoints[i].angle / 180 * 3.14)),
                 int(keypoints[i].y) + int(10 * np.sin(keypoints[i].angle / 180 * 3.14)))
        cv2.line(img1, ptStart, ptEnd, (0, 0, 255), 2, 4)

    cv2.imshow("image", img1)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()


# import cv2 as cv
# from sift import *
# import random
#
#
# def matcher(desc1, desc2, thr):
#     matchs = []
#     minm = 100
#     for i in range(len(desc1)):
#         for j in range(len(desc2)):
#             dst = np.sqrt(np.sum((desc1[i]['hist'] - desc2[j]['hist']) ** 2))
#             minm = np.min([minm, dst])
#             if dst < thr:
#                 matchs.append([i, j])
#     print(minm)
#     return matchs
#
#
# input = 'lena1.jpg'
# img = cv.imread(input, 0)
# img1 = cv.imread(input)
# start = time.time()
# s = SIFT(nfeatures=0,
#          edgeThreshold=10,
#          contrastThreshold=0.04)
# keypoints1, descriptors1, boxes = s.run_sift_feaures(img)
# end = time.time()
# print(end - start, 's')
# for i in range(len(keypoints1)):
#     cv.circle(img1, (int(keypoints1[i].x), int(keypoints1[i].y)), int(keypoints1[i].size), (0, 255, 0), 1)
#     ptStart = (int(keypoints1[i].x), int(keypoints1[i].y))
#     ptEnd = (int(keypoints1[i].x) + int(20 * np.cos(keypoints1[i].angle / 180 * 3.14)),
#              int(keypoints1[i].y) - int(20 * np.sin(keypoints1[i].angle / 180 * 3.14)))
#     cv.line(img1, ptStart, ptEnd, (0, 0, 255), 2, 4)
#
# input = 'lena2.jpg'
# img = cv.imread(input, 0)
# img2 = cv.imread(input)
# start = time.time()
# s = SIFT(nfeatures=0,
#          edgeThreshold=10,
#          contrastThreshold=0.04)
# keypoints2, descriptors2, boxes = s.run_sift_feaures(img)
#
# end = time.time()
#
# for i in range(len(keypoints2)):
#     cv.circle(img2, (int(keypoints2[i].x), int(keypoints2[i].y)), int(keypoints2[i].size), (0, 255, 0), 1)
#     ptStart = (int(keypoints2[i].x), int(keypoints2[i].y))
#     ptEnd = (int(keypoints2[i].x) + int(20 * np.cos(keypoints2[i].angle / 180 * 3.14)),
#              int(keypoints2[i].y) - int(20 * np.sin(keypoints2[i].angle / 180 * 3.14)))
#     cv.line(img2, ptStart, ptEnd, (0, 0, 255), 1, 4)
#
# print(end - start, 's')
# matchs = matcher(descriptors1, descriptors2, 0.25)
# print(len(matchs))
#
# col = np.max([img1.shape[0], img2.shape[0]])
# row = img1.shape[1] + img2.shape[1]
# channel = 3
# image_mathch = np.zeros((col, row, channel), dtype=np.uint8)
# image_mathch[:img1.shape[0], :img1.shape[1], :] = img1
# image_mathch[:img2.shape[0], img1.shape[1]:, :] = img2
# for i in range(len(matchs)):
#     b = random.randint(0, 255)
#     g = random.randint(0, 255)
#     r = random.randint(0, 255)
#     point_color = (b, g, r)
#     x1 = int(descriptors1[matchs[i][0]]['x'])
#     y1 = int(descriptors1[matchs[i][0]]['y'])
#     ptStart = (x1, y1)
#     x2 = int(descriptors2[matchs[i][1]]['x']) + img1.shape[0]
#     y2 = int(descriptors2[matchs[i][1]]['y'])
#     ptEnd = (x2, y2)
#     cv2.line(image_mathch, ptStart, ptEnd, point_color, 1, 4)
#
# cv.imshow("image", image_mathch)
# cv.waitKey(0)
