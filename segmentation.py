# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'segmentation.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 400)
        MainWindow.setMinimumSize(QtCore.QSize(640, 400))
        MainWindow.setMaximumSize(QtCore.QSize(640, 400))
        MainWindow.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.showup = QtWidgets.QFrame(self.centralwidget)
        self.showup.setGeometry(QtCore.QRect(0, 0, 542, 276))
        self.showup.setMinimumSize(QtCore.QSize(542, 276))
        self.showup.setMaximumSize(QtCore.QSize(542, 276))
        self.showup.setObjectName("showup")
        self.image_views = QtWidgets.QHBoxLayout(self.showup)
        self.image_views.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.image_views.setContentsMargins(10, 20, 10, 0)
        self.image_views.setSpacing(1)
        self.image_views.setObjectName("image_views")
        self.origin = QtWidgets.QLabel(self.showup)
        self.origin.setMinimumSize(QtCore.QSize(256, 256))
        self.origin.setMaximumSize(QtCore.QSize(256, 256))
        self.origin.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.origin.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.origin.setLineWidth(1)
        self.origin.setText("")
        self.origin.setTextFormat(QtCore.Qt.AutoText)
        self.origin.setObjectName("origin")
        self.image_views.addWidget(self.origin)
        self.terminal = QtWidgets.QLabel(self.showup)
        self.terminal.setMinimumSize(QtCore.QSize(256, 256))
        self.terminal.setMaximumSize(QtCore.QSize(256, 256))
        self.terminal.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.terminal.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.terminal.setLineWidth(1)
        self.terminal.setText("")
        self.terminal.setTextFormat(QtCore.Qt.AutoText)
        self.terminal.setObjectName("terminal")
        self.image_views.addWidget(self.terminal)
        self.dataset = QtWidgets.QFrame(self.centralwidget)
        self.dataset.setGeometry(QtCore.QRect(0, 300, 600, 83))
        self.dataset.setMinimumSize(QtCore.QSize(600, 83))
        self.dataset.setMaximumSize(QtCore.QSize(600, 83))
        self.dataset.setObjectName("dataset")
        self.choices = QtWidgets.QVBoxLayout(self.dataset)
        self.choices.setContentsMargins(5, 5, 5, 5)
        self.choices.setSpacing(0)
        self.choices.setObjectName("choices")
        self.toseg = QtWidgets.QHBoxLayout()
        self.toseg.setContentsMargins(5, 5, 5, 5)
        self.toseg.setSpacing(1)
        self.toseg.setObjectName("toseg")
        self.ttext = QtWidgets.QPushButton(self.dataset)
        self.ttext.setMinimumSize(QtCore.QSize(81, 25))
        self.ttext.setMaximumSize(QtCore.QSize(81, 25))
        self.ttext.setObjectName("ttext")
        self.toseg.addWidget(self.ttext)
        self.tpath = QtWidgets.QLineEdit(self.dataset)
        self.tpath.setMinimumSize(QtCore.QSize(419, 25))
        self.tpath.setMaximumSize(QtCore.QSize(419, 25))
        self.tpath.setReadOnly(True)
        self.tpath.setObjectName("tpath")
        self.toseg.addWidget(self.tpath)
        self.tbutton = QtWidgets.QPushButton(self.dataset)
        self.tbutton.setMinimumSize(QtCore.QSize(64, 25))
        self.tbutton.setMaximumSize(QtCore.QSize(64, 25))
        self.tbutton.setObjectName("tbutton")
        self.toseg.addWidget(self.tbutton)
        self.choices.addLayout(self.toseg)
        self.standard = QtWidgets.QHBoxLayout()
        self.standard.setContentsMargins(5, 5, 5, 5)
        self.standard.setSpacing(1)
        self.standard.setObjectName("standard")
        self.stext = QtWidgets.QPushButton(self.dataset)
        self.stext.setMinimumSize(QtCore.QSize(81, 25))
        self.stext.setMaximumSize(QtCore.QSize(81, 25))
        self.stext.setObjectName("stext")
        self.standard.addWidget(self.stext)
        self.spath = QtWidgets.QLineEdit(self.dataset)
        self.spath.setMinimumSize(QtCore.QSize(419, 25))
        self.spath.setMaximumSize(QtCore.QSize(419, 25))
        self.spath.setReadOnly(True)
        self.spath.setObjectName("spath")
        self.standard.addWidget(self.spath)
        self.sbutton = QtWidgets.QPushButton(self.dataset)
        self.sbutton.setMinimumSize(QtCore.QSize(64, 25))
        self.sbutton.setMaximumSize(QtCore.QSize(64, 25))
        self.sbutton.setObjectName("sbutton")
        self.standard.addWidget(self.sbutton)
        self.choices.addLayout(self.standard)
        self.placeholder = QtWidgets.QFrame(self.centralwidget)
        self.placeholder.setGeometry(QtCore.QRect(11, 292, 522, 5))
        self.placeholder.setMinimumSize(QtCore.QSize(522, 5))
        self.placeholder.setMaximumSize(QtCore.QSize(522, 5))
        self.placeholder.setLineWidth(3)
        self.placeholder.setFrameShape(QtWidgets.QFrame.HLine)
        self.placeholder.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.placeholder.setObjectName("placeholder")
        self.dtext = QtWidgets.QLabel(self.centralwidget)
        self.dtext.setGeometry(QtCore.QRect(540, 30, 41, 31))
        self.dtext.setMinimumSize(QtCore.QSize(41, 31))
        self.dtext.setMaximumSize(QtCore.QSize(41, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.dtext.setFont(font)
        self.dtext.setObjectName("dtext")
        self.dpecent = QtWidgets.QLabel(self.centralwidget)
        self.dpecent.setGeometry(QtCore.QRect(600, 70, 31, 21))
        self.dpecent.setMinimumSize(QtCore.QSize(31, 21))
        self.dpecent.setMaximumSize(QtCore.QSize(31, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.dpecent.setFont(font)
        self.dpecent.setObjectName("dpecent")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(540, 210, 92, 62))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.glance = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.glance.setContentsMargins(0, 0, 0, 0)
        self.glance.setSpacing(10)
        self.glance.setObjectName("glance")
        self.last = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.last.setMinimumSize(QtCore.QSize(90, 25))
        self.last.setMaximumSize(QtCore.QSize(90, 25))
        self.last.setObjectName("last")
        self.glance.addWidget(self.last)
        self.next = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.next.setMinimumSize(QtCore.QSize(90, 25))
        self.next.setMaximumSize(QtCore.QSize(90, 25))
        self.next.setObjectName("next")
        self.glance.addWidget(self.next)
        self.dice = QtWidgets.QLabel(self.centralwidget)
        self.dice.setGeometry(QtCore.QRect(563, 60, 40, 30))
        self.dice.setMinimumSize(QtCore.QSize(40, 30))
        self.dice.setMaximumSize(QtCore.QSize(40, 30))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.dice.setFont(font)
        self.dice.setObjectName("dice")
        self.mdtext = QtWidgets.QLabel(self.centralwidget)
        self.mdtext.setGeometry(QtCore.QRect(540, 95, 80, 31))
        self.mdtext.setMinimumSize(QtCore.QSize(80, 31))
        self.mdtext.setMaximumSize(QtCore.QSize(41, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.mdtext.setFont(font)
        self.mdtext.setObjectName("mdtext")
        self.mdice = QtWidgets.QLabel(self.centralwidget)
        self.mdice.setGeometry(QtCore.QRect(563, 125, 40, 30))
        self.mdice.setMinimumSize(QtCore.QSize(40, 30))
        self.mdice.setMaximumSize(QtCore.QSize(40, 30))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.mdice.setFont(font)
        self.mdice.setObjectName("mdice")
        self.mdpecent = QtWidgets.QLabel(self.centralwidget)
        self.mdpecent.setGeometry(QtCore.QRect(600, 135, 31, 21))
        self.mdpecent.setMinimumSize(QtCore.QSize(31, 21))
        self.mdpecent.setMaximumSize(QtCore.QSize(31, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.mdpecent.setFont(font)
        self.mdpecent.setObjectName("mdpecent")
        self.dura_text = QtWidgets.QLabel(self.centralwidget)
        self.dura_text.setGeometry(QtCore.QRect(540, 170, 56, 25))
        self.dura_text.setMinimumSize(QtCore.QSize(56, 25))
        self.dura_text.setMaximumSize(QtCore.QSize(56, 25))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.dura_text.setFont(font)
        self.dura_text.setObjectName("dura_text")
        self.duration = QtWidgets.QLabel(self.centralwidget)
        self.duration.setGeometry(QtCore.QRect(600, 170, 40, 30))
        self.duration.setMinimumSize(QtCore.QSize(40, 30))
        self.duration.setMaximumSize(QtCore.QSize(40, 30))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.duration.setFont(font)
        self.duration.setObjectName("duration")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.tbutton.clicked.connect(MainWindow.to_seg_path)
        self.sbutton.clicked.connect(MainWindow.standard_path)
        self.last.clicked.connect(MainWindow.last_pic)
        self.next.clicked.connect(MainWindow.next_pic)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ttext.setText(_translate("MainWindow", "待分割数据"))
        self.tbutton.setText(_translate("MainWindow", "浏览..."))
        self.stext.setText(_translate("MainWindow", "标准化结果"))
        self.sbutton.setText(_translate("MainWindow", "浏览..."))
        self.dtext.setText(_translate("MainWindow", "Dice :"))
        self.dpecent.setText(_translate("MainWindow", "%"))
        self.last.setText(_translate("MainWindow", "上一张"))
        self.next.setText(_translate("MainWindow", "下一张"))
        self.dice.setText(_translate("MainWindow", "99.99"))
        self.mdtext.setText(_translate("MainWindow", "Mean Dice :"))
        self.mdice.setText(_translate("MainWindow", "99.99"))
        self.mdpecent.setText(_translate("MainWindow", "%"))
        self.dura_text.setText(_translate("MainWindow", "执行时间："))
        self.duration.setText(_translate("MainWindow", "0.00 s"))