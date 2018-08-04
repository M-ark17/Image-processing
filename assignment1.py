#!/usr/bin/python

import sys
import PyQt4
from PyQt4 import QtGui, QtCore

class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window,self).__init__()
        self.setGeometry(50,50,1400,650)
        self.setWindowTitle("Basic Image Editor")
        self.home()

    def home(self):
        btn = QtGui.QPushButton("Upload Image",self)
        btn.clicked.connect(self.file_open)
        btn.resize(200,40)
        btn.move(500,50 )
        btn1 = QtGui.QPushButton("Equalize histogram",self)
        btn1.clicked.connect(self.save_image)
        btn1.resize(200,40)
        btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        btn1.move(500,100 )
        btn2 = QtGui.QPushButton("Gamma correct",self)
        btn2.clicked.connect(self.win_close)
        btn2.resize(200,40)
        btn2.move(500,150 )
        btn3 = QtGui.QPushButton("Log transform",self)
        btn3.clicked.connect(self.win_close)
        btn3.resize(200,40)
        btn3.move(500,200 )
        btn4 = QtGui.QPushButton("Blur Image",self)
        btn4.clicked.connect(self.win_close)
        btn4.resize(200,40)
        btn4.move(500,250 )
        btn5 = QtGui.QPushButton("Sharpening",self)
        btn5.clicked.connect(self.win_close)
        btn5.resize(200,40)
        btn5.move(500,300 )
        btn6 = QtGui.QPushButton("Additional Feature",self)
        btn6.clicked.connect(self.save_image)
        btn6.resize(200,40)
        btn6.move(500,350 )
        btn7 = QtGui.QPushButton("Undo last Change",self)
        btn7.clicked.connect(self.save_image)
        btn7.resize(200,40)
        btn7.move(500,400 )
        btn8 = QtGui.QPushButton("Undo All Changes",self)
        btn8.clicked.connect(self.save_image)
        btn8.resize(200,40)
        btn8.move(500,450 )
        btn9 = QtGui.QPushButton("Save Image",self)
        btn9.clicked.connect(self.save_image)
        btn9.resize(200,40)
        btn9.move(500,500 )
        btn10 = QtGui.QPushButton("Close Window",self)
        btn10.clicked.connect(self.win_close)
        btn10.resize(200,40)
        btn10.move(500,550 )


        self.show()

    def file_open(self):
        name = QtGui.QFileDialog.getOpenFileName(self,'Open File','','Images (*.png *.xpm *.jpg *.jpeg)')
        file = open(name,'r')
        # with file:

    def save_image(self):
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File','','Images (*.png *.xpm *.jpg *.jpeg)')
        file = open(name,'w')
        file.close()

    def win_close(self):
        print("Window closed")
        sys.exit()


def main():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())
main()
