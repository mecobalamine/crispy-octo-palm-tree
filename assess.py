import os
import const
import cv2 as cv
import time
import numpy as np
import argparse

from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
from dummy import merge
from morphsnakes import dummy_image, dummy_level_set, MorphACWE


def segment(origin_file, terminal_file):
    image = cv.imread(origin_file, cv.IMREAD_GRAYSCALE)
    origin_image = image.copy()
    terminal = cv.imread(terminal_file, cv.IMREAD_GRAYSCALE)
    terminal = get_mask(terminal)

    start_time = time.time()

    # segment
    ms = MorphACWE(image, smoothing=const.smoothing(), lambda1=const.lambda1(),
                   lambda2=const.lambda2(), iteration=const.iteration())

    # gamma = checkerboard_level_set(image.shape)
    gamma = dummy_level_set(image)
    thr_res = gamma

    ms.run(gamma=gamma)

    end_time = time.time()

    origin = ms.gamma
    # fig = plt.figure()
    # fig.clf()
    # ax0 = fig.add_subplot(1, 1, 1)
    # ax0.imshow(origin, cmap='gray')
    # ax0.set_axis_off()
    # plt.show()

    origin = merge(origin, thr_res)

    result = get_result(image, origin)

    op_time = '%.2f' % (end_time - start_time)

    return assess(origin, terminal), origin_image, result, op_time


def get_mask(image):
    shape = image.shape
    height = shape[0]
    width = shape[1]

    mask = np.zeros(shape)
    for y in range(height):
        for x in range(width):
            if image[y, x] != 0:
                mask[y, x] = 1
    return mask


def get_result(image, mask):
    shape = image.shape
    height = shape[0]
    width = shape[1]

    result = np.zeros(shape, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 1:
                result[y, x] = image[y, x]
            # else:
            #     result[y, x] = 255
    return result


def assess(origin, terminal):
    shape = origin.shape
    height = shape[0]
    width = shape[1]

    # if mask is None:
    #     mask = np.equal(origin, terminal)

    # category = 2
    # union = np.array([0] * category)
    # cross = np.array([0] * category)
    # target = np.array([0] * category)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # calculate the union area and intersection area
    for y in range(height):
        for x in range(width):
            label = int(origin[y, x])
            clazz = int(terminal[y, x])

            mask = (label == clazz)

            if mask:
                if clazz == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if clazz == 1:
                    fp += 1
                else:
                    fn += 1

    # fpr = float(fp / (tp + fp + fn))
    # print('过分割%.4f' % fpr)
    #
    # fnr = float(fn / (tp + fp + fn))
    # print('欠分割%.4f' % fnr)

    # print('pa%.4f' % float((tp + tn) / (tp + tn + fp + fn)))
    # print('iou%.4f' % float(tp / (tp + fn + fp)))

    dice = float(tp * 2 / (tp * 2 + fn + fp))
    # print('dice%.4f' % dice)

    # target[clazz] += 1
    # target[label] += 1
    #
    # # without considering those uncertain boundaries
    # if mask:
    #     cross[clazz] += 1
    #     union[clazz] += 1
    # else:
    #     union[clazz] += 1
    #     union[label] += 1
    # calculate results of each class
    # for i, (c, u) in enumerate(zip(cross, union)):
    #     print('%5d:%10d  /%10d\t\t%.4f' % (i, c, u, c / u if u != 0 else 0))

    # avg = sum(c for c in cross) / sum(u for u in union)
    # print('PA:%.4f' % avg)
    #
    # avg = sum([c / u if u != 0 else 0 for c, u in
    #            zip(cross, union)]) / category
    # print('MPA:%.4f' % avg)
    #
    # avg = sum([c / (u + u - c) if u != 0 else 0 for c, u, t in
    #            zip(cross, union, target)]) / category
    # print('MIoU:%.4f' % avg)
    #
    return dice


def mpa(category, origin, mask=None):
    union, cross, target = assess(category, origin)
    avg = sum(c for c in cross) / sum(t for t in target)
    return avg


def pa(category, origin, mask=None):
    union, cross, target = assess(category, origin)
    avg = sum([c / u if u != 0 else 0 for c, u in
               zip(cross, union)]) / category
    return avg


def miou(category, origin, mask=None):
    union, cross, target = assess(category, origin)
    avg = sum([c / (u + t - c) if u != 0 else 0 for c, u, t in
               zip(cross, union, target)]) / category
    return avg


if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--check_file", default='check_list.txt', type=str)
    # parser.add_argument("--label_dir", default='label_data', type=str)
    # parser.add_argument("--check_dir", default='generated_data', type=str)
    # args = parser.parse_args()
    # label_dir = args.label_dir
    # check_dir = args.check_dir
    # check_file = args.check_file
    # # some predefined param shadows
    # cal_list = [string.replace('\n', '') for string in open(check_file).readlines()]
    # c_num = 2
    # success_num = 0
    # avg = 0
    # union_cnt = np.array([0] * 2)
    # intersection_cnt = np.array([0] * 2)
    #
    # # begin evaluation
    # for i, name in enumerate(cal_list):
    #     print('\n{}/{}:{}'.format(i + 1, len(cal_list), name))
    #
    #     # get label_img and check img
    #     if not os.path.exists(os.path.join(label_dir, name + '.png')):
    #         print('label img not found')
    #         continue
    #     label_img = cv.imread(const.input_file).reshape(-1)
    #     if not os.path.exists(os.path.join(check_dir, name + '.png')):
    #         print('check img not found ')
    #         continue
    #     check_img = cv.imread(os.path.join(check_dir, name + '.png'))[:, :, 0].reshape(-1)
    #     success_num += 1
    #
    #     # calculate the union area and intersection area
    #     mask_img = np.equal(check_img, label_img)
    #     for j in range(len(label_img)):
    #         c_value = check_img[j]
    #         l_value = label_img[j]
    #         # without considering those uncertain boundaries
    #         if l_value == 255:
    #             continue
    #         if mask_img[j]:
    #             intersection_cnt[c_value] += 1
    #             union_cnt[c_value] += 1
    #         else:
    #             union_cnt[c_value] += 1
    #             union_cnt[l_value] += 1
    #     # calculate results of each class
    #     for j, (intersection, union) in enumerate(zip(intersection_cnt, union_cnt)):
    #         print('%5d:%10d  /%10d\t\t%.4f' % (j, intersection, union, intersection / union if union != 0 else 0))
    #     avg = sum([intersection / union if union != 0 else 0 for intersection, union in
    #                zip(intersection_cnt, union_cnt)]) / c_num
    #     print('avg:%.4f' % avg)
    #
    # print('\n\n finish with %d/%d\n the MIoU:%.4lf' % (success_num, len(cal_list), avg))
