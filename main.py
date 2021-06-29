import sys
import time
import datetime
import os
import numpy as np

from PyQt5.QtGui import QImage, QPixmap

from os import listdir
from os.path import isfile, join, splitext
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QGraphicsPixmapItem, QMessageBox

from segmentation import Ui_MainWindow
from assess import segment

from morphsnakes import MorphACWE, dummy_image, dummy_level_set


def is_img(file):
    ext = splitext(file)[1]
    ext = ext.lower()
    if ext in ['.jpg', '.png', '.jpeg', '.bmp']:
        return True
    else:
        return False


def check(o_files, t_files):
    if o_files is None or t_files is None:
        return False
    if len(o_files) != len(t_files):
        return False
    for of, tf in zip(o_files, t_files):
        if of != tf:
            return False
        elif not is_img(of):
            return False
    return True


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.shape = [256, 256]

        self.origin_path = None
        self.terminal_path = None
        self.files = None

        self.curr = 0
        self.count = 0
        self.length = 0
        self.dice_sum = 0

        self.dices = None
        self.to_seg = None
        self.result = None
        self.op_times = None

    def to_seg_path(self):
        if self.files is not None:
            self.clear()

        path = QFileDialog.getExistingDirectory(self, "请选择文件夹路径", "./")
        self.origin_path = path

        self.tpath.setText(path)
        QtWidgets.QApplication.processEvents()

        if self.terminal_path is not None:
            self.start()

    def standard_path(self):
        if self.files is not None:
            self.clear()

        path = QFileDialog.getExistingDirectory(self, "请选择文件夹路径", "./")
        self.terminal_path = path

        self.spath.setText(path)
        QtWidgets.QApplication.processEvents()

        if self.origin_path is not None:
            self.start()

    def start(self):
        origin_path = self.origin_path
        terminal_path = self.terminal_path
        files = [f for f in listdir(origin_path) if isfile(join(origin_path, f))]
        t_files = [f for f in listdir(terminal_path) if isfile(join(terminal_path, f))]

        if not check(files, t_files):
            msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'Wrong input files!')
            msg_box.exec_()
            self.clear()
            return

        length = len(files)
        self.length = length
        self.dices = [''] * length
        self.to_seg = [np.zeros(self.shape)] * length
        self.result = [np.zeros(self.shape)] * length
        self.op_times = [''] * length

        self.files = files

        origin_file = os.path.join(origin_path, files[0])
        terminal_file = os.path.join(terminal_path, files[0])
        self.transit(origin_file, terminal_file)

    def last_pic(self):
        if self.length == 0:
            return

        curr = self.curr
        if curr > 0:
            curr -= 1
        else:
            curr = self.count - 1
        self.curr = curr

        op_time = self.op_times[curr]
        dice = self.dices[curr]
        origin = self.to_seg[curr]
        terminal = self.result[curr]
        # print("last" + str(self.curr))
        self.change(op_time, origin, terminal, dice)

    def next_pic(self):
        if self.length == 0:
            return

        curr = self.curr + 1
        if curr < self.count:
            self.curr = curr

            op_time = self.op_times[curr]
            dice = self.dices[curr]
            origin = self.to_seg[curr]
            terminal = self.result[curr]
            self.change(op_time, origin, terminal, dice)

        elif curr == self.length:
            self.curr = 0
            op_time = self.op_times[0]
            dice = self.dices[0]
            origin = self.to_seg[0]
            terminal = self.result[0]
            self.change(op_time, origin, terminal, dice)

        else:
            self.curr = curr

            origin_file = os.path.join(self.origin_path, self.files[curr])
            terminal_file = os.path.join(self.terminal_path, self.files[curr])
            self.transit(origin_file, terminal_file)

        # print("next" + str(self.curr))

    def transit(self, origin_file, terminal_file):
        self.next.setEnabled(False)

        dice, origin, terminal, op_time = segment(origin_file, terminal_file)

        self.dice_sum += dice
        str_dice = '%.2f' % (dice * 100)

        count = self.count
        mdice = '%.2f' % (self.dice_sum / (count + 1) * 100)

        self.change(op_time, origin, terminal, str_dice, mdice)

        self.next.setEnabled(True)

        self.dices[count] = str_dice
        self.to_seg[count] = origin
        self.result[count] = terminal
        self.op_times[count] = op_time
        self.count = count + 1

    def change(self, op_time, origin, terminal, dice, mdice=None):
        shape = self.shape
        origin_frame = QImage(origin, shape[1], shape[0], QImage.Format_Grayscale8)
        origin_pixmap = QPixmap.fromImage(origin_frame)
        self.origin.setPixmap(origin_pixmap)
        QtWidgets.QApplication.processEvents()

        terminal_frame = QImage(terminal.data, shape[1], shape[0], QImage.Format_Grayscale8)
        terminal_pixmap = QPixmap.fromImage(terminal_frame)
        self.terminal.setPixmap(terminal_pixmap)
        QtWidgets.QApplication.processEvents()

        self.dice.setText(dice)
        QtWidgets.QApplication.processEvents()

        if mdice is not None:
            self.mdice.setText(mdice)
            QtWidgets.QApplication.processEvents()

        self.duration.setText(op_time + ' s')
        QtWidgets.QApplication.processEvents()

    def clear(self):
        self.origin_path = None
        self.terminal_path = None
        self.files = None

        self.curr = 0
        self.count = 0
        self.length = 0
        self.dice_sum = 0

        self.dices = None
        self.to_seg = None
        self.result = None
        self.op_times = None

        self.tpath.setText('')
        self.spath.setText('')
        self.origin.setPixmap(QPixmap(''))
        self.terminal.setPixmap(QPixmap(''))
        self.dice.setText('99.99')
        self.mdice.setText('99.99')
        self.duration.setText('0.00 s')


if __name__ == '__main__':
    # QApplication相当于main函数，也就是整个程序（很多文件）的主入口函数。
    # 对于GUI程序必须至少有一个这样的实例来让程序运行。
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # 解决了QtDesigner设计的界面与实际运行界面不一致的问题
    app = QtWidgets.QApplication(sys.argv)
    # 生成 MainWindow 类的实例。
    window = MainWindow()
    # 有了实例，就得让它显示，show()是QWidget的方法，用于显示窗口。
    window.show()
    # 调用sys库的exit退出方法，条件是app.exec_()，也就是整个窗口关闭。
    # 有时候退出程序后，sys.exit(app.exec_())会报错，改用app.exec_()就没事
    sys.exit(app.exec_())
