import cv2 as cv


class Const:
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("can't change const %s" % name)
        if not name.isupper():
            raise self.ConstCaseError('const name "%s" is not all uppercase' % name)
        self.__dict__[name] = value


const = Const()

const.INPUT_FILE = "./assets/seg_test.jpg"
const.OUTPUT_FILE = "./assets/seg_test_out_graph.jpg"
const.RESULT_FILE = "./assets/seg_test_result.jpg"

const.INPUT_PATH = "./assets/images"
const.OUTPUT_PATH = "./assets/results"

const.SMOOTHING = 4
const.LAMBDA1 = 1
const.LAMBDA2 = 5
const.ITERATION = 5

const.KERNEL = (5, 5)


def input_file():
    return const.INPUT_FILE


def output_file():
    return const.OUTPUT_FILE


def result_file():
    return const.RESULT_FILE


def kernel():
    return const.KERNEL


def smoothing():
    return const.SMOOTHING


def lambda1():
    return const.LAMBDA1


def lambda2():
    return const.LAMBDA2


def iteration():
    return const.ITERATION


def get_target():
    target = cv.imread(result_file(), cv.IMREAD_GRAYSCALE)
    shape = target.shape
    height = shape[0]
    width = shape[1]
    for y in range(height):
        for x in range(width):
            if target[y, x] != 0:
                target[y, x] = 1
    return target
