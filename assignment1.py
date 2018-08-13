#!/usr/bin/python

import sys
import PyQt4
import math
import numpy as np
import cv2 as cv
from PyQt4 import QtGui, QtCore

class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window,self).__init__()
        self.setGeometry(50,50,1400,650)
        self.setWindowTitle("Basic Image Editor")
        self.home()
        self.__pixmap = None
        self.__mdfd_img_lstchg = None
        self.__mdfd_img = None
        self.__img_h = None
        self.__img_s = None
        self.__img_v = None
        self.__img_height = None
        self.__img_width = None
        self.lbl = QtGui.QLabel(self)
        self.lbl1 = QtGui.QLabel(self)
        self.lbl2 = QtGui.QLabel(self)
        self.lbl3 = QtGui.QLabel(self)
        self.lbl_s1 = QtGui.QLabel(self)
        self.lbl_s2 = QtGui.QLabel(self)
        self.s1 = QtGui.QScrollBar(self)
        self.s2 = QtGui.QScrollBar(self)

    def home(self):
        btn = QtGui.QPushButton("Upload Image",self)
        # image = btn.clicked.connect(self.file_open)
        btn.clicked.connect(self.file_open)
        btn.resize(200,40)
        btn.move(500,50 )
        btn1 = QtGui.QPushButton("Equalize histogram",self)
        btn1.clicked.connect(self.hist_equal)
        btn1.resize(200,40)
        btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        btn1.move(500,100 )
        btn2 = QtGui.QPushButton("Gamma correct",self)
        btn2.clicked.connect(self.gamma_correct_btn)
        btn2.resize(200,40)
        btn2.move(500,150 )
        btn3 = QtGui.QPushButton("Log transform",self)
        btn3.clicked.connect(self.log_transform)
        btn3.resize(200,40)
        btn3.move(500,200 )
        btn4 = QtGui.QPushButton("Blur Image",self)
        btn4.clicked.connect(self.blur_img_scr_bar)
        btn4.resize(200,40)
        btn4.move(500,250 )
        btn5 = QtGui.QPushButton("Sharpening",self)
        btn5.clicked.connect(self.sharpen_img_scr_bar)
        btn5.resize(200,40)
        btn5.move(500,300 )
        btn6 = QtGui.QPushButton("Additional Feature",self)
        btn6.clicked.connect(self.save_image)
        btn6.resize(200,40)
        btn6.move(500,350 )
        btn7 = QtGui.QPushButton("Undo last Change",self)
        btn7.clicked.connect(self.undo)
        btn7.resize(200,40)
        btn7.move(500,400 )
        btn8 = QtGui.QPushButton("Undo All Changes",self)
        btn8.clicked.connect(self.undoall)
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
        upld_img = QtGui.QImage()
        self.__ip_img =  cv.imread(str(name),cv.IMREAD_COLOR)
        img_hsv = cv.cvtColor(self.__ip_img, cv.COLOR_BGR2HSV)
        # get image properties.
        self.__img_h,self.__img_s,self.__img_v = cv.split(img_hsv)
        self.__img_height,self.__img_width = self.__img_v.shape
        # print self.__img_v.shape
        self.__mdfd_img_lstchg = self.__img_v
        self.__mdfd_img = self.__img_v
        if upld_img.load(name):
            self.lbl1.clear()
            self.lbl1.show()
            self.lbl1.setText("Orignal Image")
            self.lbl1.move(200,50)
            self.lbl1.show()
            pixmap = QtGui.QPixmap(upld_img)
            self.__pixmap = pixmap.scaled(400, 650, QtCore.Qt.KeepAspectRatio)
            self.lbl.clear()
            self.lbl.show()
            self.lbl.resize(400,650)
            self.lbl.move(50,0)
            self.lbl.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
            self.lbl.setScaledContents(False)
            self.lbl.setPixmap(self.__pixmap)
            self.lbl.show()
            print("Selected Image uploaded")

        else:
            print("Could not upload Image")

    def hist_equal(self):
        self.__mdfd_img_lstchg = self.__mdfd_img
        sum = 0
        hist_equal_img = self.__mdfd_img
        new_img=np.empty_like(hist_equal_img)
        # print(np.max(hist_equal_img))
        for i in range (256):
            idx = np.where(hist_equal_img == i)
            i_intnsty_freq = len(idx[0])
            sum = sum +i_intnsty_freq
            new_intnsty = (float(sum)/hist_equal_img.size)*255.0
            # print new_intnsty
            new_img[idx] = new_intnsty
        self.__mdfd_img = new_img
        self.disp("Histogram Equalization")
        print("Histogram Equalized")

    def gamma_correct_btn(self):
        gamma,ok = QtGui.QInputDialog.getDouble(self,"integer input dialog","enter a number")
        if ok:
            print 'Gamma value = '+str(gamma)
            self.gamma_correct(gamma)
        else:
            print("No input gamma value given")

    def gamma_correct(self,gamma):
        self.__mdfd_img_lstchg = self.__mdfd_img
        gamma_correct_img = self.__mdfd_img
        new_img=np.empty_like(gamma_correct_img)
        c = 1
        for i in range (256):
            idx = (gamma_correct_img == i)
            new_intnsty = c*(float(i)**gamma)
            new_img[idx] = int(new_intnsty)
        gamma_correct_img = new_img
        self.__mdfd_img = gamma_correct_img
        self.disp("Gamma transformation")
        print("Gamma transformation Applied")

    def log_transform(self):
        self.__mdfd_img_lstchg = self.__mdfd_img
        log_trnsfrm_img = self.__mdfd_img
        new_img = np.empty_like(log_trnsfrm_img)
        c = 100
        for i in range (256):
            idx = (log_trnsfrm_img == i)
            new_intnsty = float(c*(math.log10(i+1)))
            new_img[idx] = new_intnsty
        log_trnsfrm_img = new_img
        self.__mdfd_img = log_trnsfrm_img
        self.disp("Log transformation")
        print("Log transformation Applied")

    def blur_img_scr_bar(self):

        self.s1.resize(20,400)
        self.s1.move(1330,100)
        self.s1.setMaximum(255)
        self.s1.setMinimum(0)
        self.s1.show()
        self.lbl_s1.resize(30,50)
        self.lbl_s1.setText("Low")
        self.lbl_s1.move(1320,500)
        self.lbl_s1.show()
        self.lbl_s2.resize(30,50)
        self.lbl_s2.setText("High")
        self.lbl_s2.move(1320,50)
        self.lbl_s2.show()
        self.s1.valueChanged.connect(self.blur_img)

    def blur_img(self):
        print("image blurred")

    def sharpen_img_scr_bar(self):

        self.s2.resize(20,400)
        self.s2.move(1330,100)
        self.s2.setMaximum(255)
        self.s2.setMinimum(0)
        self.s2.show()
        self.lbl_s1.resize(30,50)
        self.lbl_s1.setText("Low")
        self.lbl_s1.move(1320,500)
        self.lbl_s1.show()
        self.lbl_s2.resize(30,50)
        self.lbl_s2.setText("High")
        self.lbl_s2.move(1320,50)
        self.lbl_s1.show()
        self.lbl_s2.show()
        self.s2.valueChanged.connect(self.sharpen_img)

    def sharpen_img(self):
        print("Image Sharpening Done")

    def undoall(self):
        self.__mdfd_img = self.__img_v
        # self.__mdfd_img_lstchg = self.__img_v
        self.disp("All changes undone")
        print("All changes UNDONE ")

    def undo(self):
        self.__mdfd_img = self.__mdfd_img_lstchg
        self.disp("Last change undone")
        print("Last change UNDONE ")

    def save_image(self):
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File','','Images (*.png *.xpm *.jpg *.jpeg)')
        itos = cv.merge([self.__img_h,self.__img_s, self.__mdfd_img])
        img_to_save = QtGui.QPixmap(QtGui.QImage(itos,self.__img_width, self.__img_height,3*self.__img_width, QtGui.QImage.Format_RGB888))
        if img_to_save.save(name):
            print("Image Saved To file")
        else:
            print("Could not save the Image to folder")

    def win_close(self):
        print("Window closed")
        sys.exit()

    def disp(self,txt,flag = 0):
        if (flag == 0):
            self.lbl_s1.clear()
            self.lbl_s2.clear()
            self.s1.hide()
            self.s2.hide()

        img_pix1 = cv.merge([self.__img_h,self.__img_s, self.__mdfd_img])
        img_color = cv.cvtColor(img_pix1, cv.COLOR_HSV2RGB)
        pix_img = QtGui.QPixmap(QtGui.QImage(img_color,self.__img_width, self.__img_height,3*self.__img_width, QtGui.QImage.Format_RGB888))
        self.lbl2.clear()
        self.lbl2.setText(txt)
        self.lbl2.resize(200,50)
        self.lbl2.move(950,50)
        self.lbl2.show()
        pix_img= pix_img.scaled(400, 650, QtCore.Qt.KeepAspectRatio)
        self.lbl3.clear()
        self.lbl3.resize(400,650)
        self.lbl3.move(800,0)
        self.lbl3.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.lbl3.setScaledContents(False)
        self.lbl3.setPixmap(pix_img)
        self.lbl3.show()

def main():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    # GUI.disp()
    sys.exit(app.exec_())
main()
