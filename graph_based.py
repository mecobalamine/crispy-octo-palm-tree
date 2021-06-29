import argparse
import const
import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt


class Node(object):
    def __init__(self, parent, rank=0, size=1):
        self.parent = parent
        self.rank = rank
        self.size = size

    def __repr__(self):
        return '(parent=%s, rank=%s, size=%s)' % (self.parent, self.rank, self.size)


class Forest(object):
    def __init__(self, num_of_nodes):
        self.nodes = [Node(i) for i in range(num_of_nodes)]
        self.count = num_of_nodes

    def size_of(self, i):
        return self.nodes[i].size

    def find(self, n):
        x = n
        while x != self.nodes[x].parent:
            if x < 0:
                return x
            self.nodes[x].parent = self.nodes[self.nodes[x].parent].parent
            x = self.nodes[x].parent

        return x

    def solid(self, n, parent):
        self.nodes[self.find(n)].parent = parent

    def union(self, p, q):
        if self.nodes[p].rank > self.nodes[q].rank:
            self.nodes[q].parent = p
            self.nodes[p].size += self.nodes[q].size
        else:
            self.nodes[p].parent = q
            self.nodes[q].size += self.nodes[p].size

            if self.nodes[p].rank == self.nodes[q].rank:
                self.nodes[q].rank += 1

        self.count -= 1

    def print(self):
        for node in self.nodes:
            print(node)


def outer_diff(image, x1, y1, x2, y2):
    _sum = np.sum((int(image[y1, x1]) - int(image[y2, x2])) ** 2)
    return np.sqrt(_sum)


def theta(size, k):
    return k * 1. / size


class Graph(object):
    def __init__(self, image, k=10.0, min_size=2000, neighborhood_8=True):
        self.image = image
        self.num_of_nodes = image.shape[0] * image.shape[1]
        self.k = k
        self.min_size = min_size
        self.neighborhood_8 = neighborhood_8

        self.index = None
        self.forest = None
        self.graph = None
        self.result = None
        self.dummy = None

    def create_edge(self, x1, y1, x2, y2):
        def vertex_id(x, y):
            return self.image.shape[1] * y + x

        dist = outer_diff(self.image, x1, y1, x2, y2)
        return vertex_id(x1, y1), vertex_id(x2, y2), dist

    def build_graph(self):
        graph = []
        height = self.image.shape[0]
        width = self.image.shape[1]
        for y in range(height):
            for x in range(width):
                if x > 0:
                    graph.append(self.create_edge(x, y, x - 1, y))

                if y > 0:
                    graph.append(self.create_edge(x, y, x, y - 1))

                if self.neighborhood_8:
                    if x > 0 and y > 0:
                        graph.append(self.create_edge(x, y, x - 1, y - 1))

                    if x > 0 and y < height - 1:
                        graph.append(self.create_edge(x, y, x - 1, y + 1))
        self.graph = graph
        return graph

    def merge_components(self):
        for edge in self.graph:
            root_p = self.forest.find(edge[0])
            root_q = self.forest.find(edge[1])

            if root_p != root_q and (self.forest.size_of(root_p) < self.min_size or
                                     self.forest.size_of(root_q) < self.min_size):
                self.forest.union(root_p, root_q)

    def segment(self):
        self.forest = Forest(self.num_of_nodes)

        def diff(_edge):
            return _edge[2]

        self.graph = sorted(self.graph, key=diff)
        tree_diff = [theta(1, self.k) for _ in range(self.num_of_nodes)]

        for edge in self.graph:
            root_p = self.forest.find(edge[0])
            root_q = self.forest.find(edge[1])

            if root_p != root_q and diff(edge) <= tree_diff[root_p] and diff(edge) <= tree_diff[root_q]:
                self.forest.union(root_p, root_q)
                new_root = self.forest.find(root_p)
                tree_diff[new_root] = diff(edge) + theta(self.forest.nodes[new_root].size, self.k)
        # self.generate_dummy()

    def generate_dummy(self):
        height = self.image.shape[0]
        width = self.image.shape[1]

        dummy = np.zeros((height, width), np.uint8)
        for y in range(height):
            for x in range(width):
                dummy[y, x] = self.forest.find(width * y + x)
        self.dummy = dummy
        # cv.imwrite("./assets/dummy.jpg", dummy)

    def generate_result(self):
        height = self.image.shape[0]
        width = self.image.shape[1]

        # def random_color():
        #     return int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)
        #
        # colors = [random_color() for _ in range(width * height)]

        index = 0
        indices = dict()
        result = np.zeros((height, width), np.uint8)
        for y in range(height):
            for x in range(width):
                component = self.forest.find(width * y + x)
                if component in indices:
                    result[y, x] = indices[component]
                else:
                    result[y, x] = index
                    indices[component] = index
                    index = index + 1
        self.index = index
        self.result = result
        # cv.imwrite("./assets/dummy.jpg", result)

    def dummy(self):
        self.build_graph()
        self.segment()
        self.generate_dummy()

    def run(self):
        print("Creating graph...")
        self.build_graph()
        print("Merging graph...")
        self.segment()
        self.merge_components()
        print("Visualizing segmentation...")
        self.generate_result()
        print("Number of components: {}".format(self.forest.count))


def get_args():
    parser = argparse.ArgumentParser(description='Graph-based Segmentation')
    # parser.add_argument('--input_path', type=str, default="./assets/images",
    #                     help='input path')
    # parser.add_argument('--output_path', type=str, default="./assets/results",
    #                     help='output path')
    parser.add_argument('--k', type=float, default=10.,
                        help='a constant to control the threshold function of the predicate')
    parser.add_argument('--min-size', type=int, default=2000,
                        help='a constant to remove all the components with fewer number of pixels')
    parser.add_argument('--neighborhood_8', type=bool, default=True,
                        help='choose the neighborhood format, 4 or 8')
    parser.add_argument('--kernel', type=tuple, default=(5, 5),
                        help='a tuple for the Gaussian Filter')

    args = parser.parse_args()

    return args


def main(args):
    input_file = const.input_file()
    output_file = const.output_file()
    image = cv.imread(input_file, cv.IMREAD_GRAYSCALE)

    # Gaussian Filter
    smooth_image = cv.GaussianBlur(image, const.kernel(), 0)

    graph = Graph(smooth_image, k=args.k, min_size=args.min_size, neighborhood_8=args.neighborhood_8)
    graph.run()
    result = graph.result

    fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image, cmap='gray')
    ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(result, cmap='Accent', interpolation='nearest')
    ax2.set_axis_off()

    plt.show()


if __name__ == '__main__':
    main(get_args())
