from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


def cross_product(arr1, arr2):
    """
    逆时针 > 0
    顺时针 < 0 (expected)
    共线 = 0
    """
    return arr1[0] * arr2[1] - arr1[1] * arr2[0]


def draw_nodes(node_arr, c):
    for i in range(0, len(node_arr)):
        x = [node_arr[i - 1][0], node_arr[i][0]]
        y = [node_arr[i - 1][1], node_arr[i][1]]
        plt.plot(x, y, color=c)


def main():
    node_arr = np.array([[2, 1.1], [4, 4], [2.5, 10], [5, 7], [4, 7], [6, 3], [6, 14],
                         [10, 7], [15, 9], [12, 1], [9, 6], [6, 2], [5, 3]])

    min_idx = np.argmin(node_arr[:, 1])
    start_point = node_arr[min_idx]
    pre_edge = np.array([-1, 0])
    edge = np.array([-1, 0])
    zpr = [start_point]

    for i in range(node_arr.shape[0]):
        cur_idx = (min_idx + i + 1) % (node_arr.shape[0])
        cur_edge = node_arr[cur_idx] - start_point
        a = cross_product(edge, cur_edge)
        b = cross_product(pre_edge, cur_edge)
        if a < 0 <= b:  # Q1b
            continue
        elif a < 0 and b < 0:  # Q1a
            zpr.append(node_arr[cur_idx])
        else:  # Q2a and Q2b,echo back
            while (len(zpr)) > 1:
                line_zj_cur = node_arr[cur_idx] - zpr[-1]
                line_cur_zj_1 = zpr[-2] - node_arr[cur_idx]
                if cross_product(line_zj_cur, line_cur_zj_1) < 0:
                    break
                else:
                    zpr.pop()
            zpr.append(node_arr[cur_idx])

        edge = node_arr[cur_idx] - zpr[-1]
        pre_edge = cur_edge

    plt.figure()
    draw_nodes(node_arr, 'k')
    draw_nodes(zpr, 'b')
    plt.show()


if __name__ == "__main__":
    main()
