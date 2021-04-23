import os
import const
import cv2 as cv
import numpy as np
import argparse

label_dir = 'label_data'
check_dir = 'generated_data'
check_file = ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_file", default='check_list.txt', type=str)
    parser.add_argument("--label_dir", default='label_data', type=str)
    parser.add_argument("--check_dir", default='generated_data', type=str)
    args = parser.parse_args()
    label_dir = args.label_dir
    check_dir = args.check_dir
    check_file = args.check_file
    # some predefined param shadows
    cal_list = [string.replace('\n', '') for string in open(check_file).readlines()]
    c_num = 2
    success_num = 0
    avg = 0
    union_cnt = np.array([0] * 2)
    intersection_cnt = np.array([0] * 2)

    # begin evaluation
    for i, name in enumerate(cal_list):
        print('\n{}/{}:{}'.format(i + 1, len(cal_list), name))

        # get label_img and check img
        if not os.path.exists(os.path.join(label_dir, name + '.png')):
            print('label img not found')
            continue
        label_img = cv.imread(const.input_file).reshape(-1)
        if not os.path.exists(os.path.join(check_dir, name + '.png')):
            print('check img not found ')
            continue
        check_img = cv.imread(os.path.join(check_dir, name + '.png'))[:, :, 0].reshape(-1)
        success_num += 1

        # calculate the union area and intersection area
        mask_img = np.equal(check_img, label_img)
        for j in range(len(label_img)):
            c_value = check_img[j]
            l_value = label_img[j]
            # without considering those uncertain boundaries
            if l_value == 255:
                continue
            if mask_img[j]:
                intersection_cnt[c_value] += 1
                union_cnt[c_value] += 1
            else:
                union_cnt[c_value] += 1
                union_cnt[l_value] += 1
        # calculate results of each class
        for j, (intersection, union) in enumerate(zip(intersection_cnt, union_cnt)):
            print('%5d:%10d  /%10d\t\t%.4f' % (j, intersection, union, intersection / union if union != 0 else 0))
        avg = sum([intersection / union if union != 0 else 0 for intersection, union in
                   zip(intersection_cnt, union_cnt)]) / c_num
        print('avg:%.4f' % avg)

    print('\n\n finish with %d/%d\n the MIoU:%.4lf' % (success_num, len(cal_list), avg))
