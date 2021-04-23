import random
import copy
import math
import numpy as np
import cv2 as cv

MAX_VALUE = 1e100


class Point:
    __slots__ = ["x", "y", "level", "cluster", "membership"]

    def __init__(self, cluster_center_num, x=0, y=0, level=0, cluster=0):
        self.x, self.y, self.level, self.cluster = x, y, level, cluster
        self.membership = [0.0 for _ in range(cluster_center_num)]


def generate_points(image, num_of_clusters):
    height = image.shape[0]
    width = image.shape[1]
    points = []
    for y in range(height):
        for x in range(width):
            points.append(Point(num_of_clusters, x=x, y=y, level=image[x][y]))
    return points


def inner_dist(point_a, point_b):
    distance = (point_a.x - point_b.x) * (point_a.x - point_b.x) + (point_a.y - point_b.y) * (point_a.y - point_b.y)
    return math.sqrt(distance) + (int(point_a.level) - int(point_b.level)) ** 2


def euclid_dist(point_a, point_b):
    distance = (point_a.x - point_b.x) * (point_a.x - point_b.x) + (point_a.y - point_b.y) * (point_a.y - point_b.y)
    return math.sqrt(distance)


def get_nearest_center(point, cluster_center_group):
    min_idx = point.cluster
    min_distance = MAX_VALUE
    for idx, center in enumerate(cluster_center_group):
        distance = euclid_dist(point, center)
        if distance < min_distance:
            min_distance = distance
            min_idx = idx
    return min_idx, min_distance


def init_cluster_centers(points, num_of_clusters):
    cluster_centers = [Point(num_of_clusters) for _ in range(num_of_clusters)]
    cluster_centers[0] = copy.copy(random.choice(points))
    distances = [0.0 for _ in range(len(points))]
    _sum = 0.0
    for center_idx in range(1, len(cluster_centers)):
        for pt_idx, point in enumerate(points):
            _, distances[pt_idx] = get_nearest_center(point, cluster_centers[:center_idx])
            _sum += distances[pt_idx]
        _sum *= random.random()
        for pt_idx, distance in enumerate(distances):
            _sum -= distance
            if _sum < 0:
                cluster_centers[center_idx] = copy.copy(points[pt_idx])
                break
    return cluster_centers


def fuzzy_c_means_clustering(image, points, num_of_clusters, m, max_itr):
    cluster_centers = init_cluster_centers(points, num_of_clusters)
    cluster_center_trace = [[center] for center in cluster_centers]
    tolerable_error, current_error = 10.0, MAX_VALUE
    itr = 0
    while current_error > tolerable_error and itr <= max_itr:
        for point in points:
            compute_single_membership(point, cluster_centers, m)
        current_centers = [Point(num_of_clusters) for _ in range(num_of_clusters)]
        for center_idx, center in enumerate(current_centers):
            upper_sum_x, upper_sum_y, lower_sum = 0.0, 0.0, 0.0
            for point in points:
                sum_u = pow(point.membership[center_idx], m)
                upper_sum_x += point.x * sum_u
                upper_sum_y += point.y * sum_u
                lower_sum += sum_u
            center.x = int(upper_sum_x / lower_sum)
            center.y = int(upper_sum_y / lower_sum)
            center.level = image[center.x][center.y]
        # update cluster center trace
        current_error = 0.0
        for idx, single_trace in enumerate(cluster_center_trace):
            single_trace.append(current_centers[idx])
            current_error += euclid_dist(single_trace[-1], single_trace[-2])
            cluster_centers[idx] = copy.copy(current_centers[idx])
        itr += 1
        print(str(itr) + " " + str(current_error))

    for point in points:
        max_idx, max_membership = 0, 0.0
        for idx, membership in enumerate(point.membership):
            if membership > max_membership:
                max_membership = membership
                max_idx = idx
        point.cluster = max_idx
    return cluster_centers, cluster_center_trace


def compute_single_membership(point, cluster_center_group, m):
    distances = [inner_dist(point, cluster_center_group[i]) for i in range(len(cluster_center_group))]
    for center_idx, membership in enumerate(point.membership):
        _sum = 0.0
        is_coincide = [False, 0]
        for idx, distance in enumerate(distances):
            if distance == 0:
                is_coincide[0] = True
                is_coincide[1] = idx
                break
            _sum += pow(float(distances[center_idx] / distance), 2.0 / (m - 1.0))
        if is_coincide[0]:
            if is_coincide[1] == center_idx:
                point.membership[center_idx] = 1.0
            else:
                point.membership[center_idx] = 0.0
        else:
            point.membership[center_idx] = 1.0 / _sum


def show_clustered_image(image, points, num_of_cluster):
    def random_color():
        return int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)

    colors = [random_color() for _ in range(num_of_cluster)]

    clustered_image = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    for pt_idx, point in enumerate(points):
        cluster_idx = point.cluster
        clustered_image[point.x, point.y] = colors[cluster_idx]

    return clustered_image


def main():
    input_file = "./assets/seg_test.jpg"
    output_file = "./assets/seg_test_out_fcm.jpg"
    image = cv.imread(input_file, cv.IMREAD_GRAYSCALE)

    # Gaussian Filter
    kernel = (5, 5)
    smooth_image = cv.GaussianBlur(image, kernel, 0)

    num_of_cluster = 4
    m = 2
    max_itr = 30
    points = generate_points(smooth_image, num_of_cluster)

    _, cluster_center_trace = fuzzy_c_means_clustering(smooth_image, points, num_of_cluster, m, max_itr)
    segmented_image = show_clustered_image(image, points, num_of_cluster)

    cv.imwrite(output_file, segmented_image)


if __name__ == '__main__':
    main()
