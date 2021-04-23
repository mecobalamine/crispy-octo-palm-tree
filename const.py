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

const.input_file = "./assets/seg_test.jpg"
const.output_file = "./assets/seg_test_out_graph.jpg"

const.input_path = "./assets/images"
const.output_path = "./assets/results"

const.kernel = (5, 5)
