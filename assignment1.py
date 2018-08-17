#!/usr/bin/python

import sys # import libraries needed
import PyQt4
import math
import numpy as np
import cv2 as cv
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4.QtCore import pyqtSlot,SIGNAL,SLOT

class Window(QtGui.QMainWindow): #create a class to display a window

    def __init__(self): #method to declare attributes of the class
        super(Window,self).__init__()
        self.setGeometry(50,50,1400,650)    # to set the size of the window to 1400*650
        self.setWindowTitle("Basic Image Editor") # give title to the window
        self.home() #method called home will have all the main features of the GUI
        self.__pixmap = None # create pixmap attribute to display image to GUI
        self.__mdfd_img_lstchg = None # to store the last the changed image data
        self.__mdfd_img = None # to store the current image data
        self.__img_h = None # empty attribute to store hue values of image
        self.__img_s = None # empty attribute to store saturation values of image
        self.__img_v = None # empty attribute to store intensity value of image
        self.__img_height = None # height of the Image
        self.__img_width = None # widht of the Image
        self.lbl = QtGui.QLabel(self)  # create a Qlabel object to display input image
        self.lbl1 = QtGui.QLabel(self) # create a Qlabel object to display title for input image
        self.lbl2 = QtGui.QLabel(self) # create a Qlabel object to display title for output image
        self.lbl3 = QtGui.QLabel(self) # create a Qlabel object to displat output image
        self.lbl_s1 = QtGui.QLabel(self) # create a Qlabel object to display scroll title "High"
        self.lbl_s2 = QtGui.QLabel(self) # create a Qlabel object to display scroll title "Low"
        self.lbl_s3 = QtGui.QLabel(self) # create a Qlabel object to display text for text editor
        self.s2 = QtGui.QScrollBar(self) # create a QScrollBar object to display scrollbar
        self.e2 = QtGui.QLineEdit(self) # create a QLineEdit object to display scroll title

    def home(self): # home method of the QMainWindow
        btn = QtGui.QPushButton("Upload Image",self) # button for uploading image
        btn.clicked.connect(self.file_open) # go to file_open method when clicked on Upload Image button
        btn.resize(200,40) # resize the button to the required size
        btn.move(500,50 ) # reposition the button at the required position
        btn1 = QtGui.QPushButton("Equalize histogram",self)
        btn1.clicked.connect(self.hist_equal) # go to hist_equal method when clicked on qualize histogram
        btn1.resize(200,40) # resize the button to the required size
        btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        btn1.move(500,100 )
        btn2 = QtGui.QPushButton("Gamma correct",self)
        btn2.clicked.connect(self.gamma_correct_btn) # go to gamma_correct_btn method when clicked on Gamma correct button
        btn2.resize(200,40) # resize the button to the required size
        btn2.move(500,150 ) # reposition the button at the required position
        btn3 = QtGui.QPushButton("Log transform",self)
        btn3.clicked.connect(self.log_transform) # go to log_transform method when clicked on Log transform button
        btn3.resize(200,40) # resize the button to the required size
        btn3.move(500,200 ) # reposition the button at the required position
        btn4 = QtGui.QPushButton("Blur Image",self)
        btn4.clicked.connect(self.blur_img_scr_bar) # go to blur_img_scr_bar method when clicked on Blur Image button
        btn4.resize(200,40) # resize the button to the required size
        btn4.move(500,250 ) # reposition the button at the required position
        btn5 = QtGui.QPushButton("Sharpening",self)
        btn5.clicked.connect(self.sharpen_img_scr_bar) # go to sharpen_img_scr_bar method when clicked on Sharpeninge button
        btn5.resize(200,40) # resize the button to the required size
        btn5.move(500,300 ) # reposition the button at the required position
        btn6 = QtGui.QPushButton("Additional Feature",self)
        btn6.clicked.connect(self.save_image) # go to save_image method when clicked on Additional Feature button
        btn6.resize(200,40) # resize the button to the required size
        btn6.move(500,350 ) # reposition the button at the required position
        btn7 = QtGui.QPushButton("Undo last Change",self)
        btn7.clicked.connect(self.undo) # go to undo method when clicked on Undo last Change button
        btn7.resize(200,40) # resize the button to the required size
        btn7.move(500,400 ) # reposition the button at the required position
        btn8 = QtGui.QPushButton("Undo All Changes",self)
        btn8.clicked.connect(self.undoall) # go to undoall method when clicked on Undo All Changes button
        btn8.resize(200,40) # resize the button to the required size
        btn8.move(500,450 ) # reposition the button at the required position
        btn9 = QtGui.QPushButton("Save Image",self)
        btn9.clicked.connect(self.save_image) # go to save_image method when clicked on Save Image button
        btn9.resize(200,40) # resize the button to the required size
        btn9.move(500,500 ) # reposition the button at the required position
        btn10 = QtGui.QPushButton("Close Window",self)
        btn10.clicked.connect(self.win_close) # go to win_close method when clicked on Close Windo button
        btn10.resize(200,40) # resize the button to the required size
        btn10.move(500,550 ) # reposition the button at the required position
        self.show() #show the window

    def file_open(self): #method to open file
        name = QtGui.QFileDialog.getOpenFileName(self,'Open File','','Images (*.png *.xpm *.jpg *.jpeg)') #this will open a dialog box to upload image only png,xpm,jpg,jpeg images are supported
        upld_img = QtGui.QImage() # create Qimage object to store the uploaded image data
        self.__ip_img =  cv.imread(str(name),cv.IMREAD_COLOR) # upload the image from the dialog box using imread in opencv library
        img_hsv = cv.cvtColor(self.__ip_img, cv.COLOR_BGR2HSV) #convert color image to HSV using cvtColor of opencv
        # get image properties.
        self.__img_h,self.__img_s,self.__img_v = cv.split(img_hsv)
        self.__img_height,self.__img_width = self.__img_v.shape
        # print self.__img_v.shape
        self.__mdfd_img_lstchg = self.__img_v # update the last changed value to uploaded image
        self.__mdfd_img = self.__img_v # update the current changed image to uploaded image
        if upld_img.load(name): # if the image is uploaded properly then upld_img.load will be true
            self.lbl1.clear() # clear the past content in label if any is present
            self.lbl1.setText("Orignal Image") # Set title for the input image to display
            self.lbl1.move(200,50) # position the title
            self.lbl1.show() # show the title
            pixmap = QtGui.QPixmap(upld_img) #convert the opencv image to pixmap to display it on GUI
            self.__pixmap = pixmap.scaled(400, 650, QtCore.Qt.KeepAspectRatio) # scale the pixmap to display it on GUI keep the Aspect Ratio of the original image
            self.lbl.clear() # clear the past content in label if any is present
            self.lbl.resize(400,650) # set the size of the input pixmap to 400*650
            self.lbl.move(50,0) # position the input pixmap
            self.lbl.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
            self.lbl.setScaledContents(False)
            self.lbl.setPixmap(self.__pixmap) # set the pixmap to the label
            self.lbl.show()# show the pixmap as image
            print("Selected Image uploaded") #print status to the terminal or IDE
            self.dialog = QProgressDialog('In progress please wait...','Cancel', 0, self.__img_v.size, self) # create a progress bar to display the progress of methods we use
            self.dialog.setWindowModality(Qt.WindowModal)
        else: #if the image is not uploaded then
            print("Could not upload Image") # print status to the terminal or IDE

    def hist_equal(self):# this method is for Histogram Equalization
        self.__mdfd_img_lstchg = self.__mdfd_img # store the last changed image data for undo method
        sum = 0 # initialise sum variable to store the cdf
        hist_equal_img = self.__mdfd_img # assign the image data to temporary variable to do operations on it
        new_img=np.empty_like(hist_equal_img) # create a empty numpy array with the same size of the input image
        # print(np.max(hist_equal_img))
        for i in range (256): # for each intensity value
            idx = np.where(hist_equal_img == i) # get the indexes of the pixels with that image intensites
            i_intnsty_freq = len(idx[0]) # find the number of such pixels
            sum = sum +i_intnsty_freq # find the CDF
            new_intnsty = (float(sum)/hist_equal_img.size)*255.0 # calculate the new intensity
            # print new_intnsty
            new_img[idx] = new_intnsty # assign the new intensity to new array
        self.__mdfd_img = new_img # now move the data in temp variable to global variable
        self.disp("Histogram Equalization") # to display the changed image
        print("Histogram Equalized") # Print status to terminal or IDE

    def gamma_correct_btn(self):
        gamma,ok = QtGui.QInputDialog.getDouble(self,"integer input dialog","enter a number")
        if ok:
            print 'Gamma value = '+str(gamma) # Print status to terminal or IDE
            self.gamma_correct(gamma)
        else:
            print("No input gamma value given") # Print status to terminal or IDE

    def gamma_correct(self,gamma):
        self.__mdfd_img_lstchg = self.__mdfd_img # store the last changed image data for undo method
        gamma_correct_img = self.__mdfd_img
        new_img_2=np.empty_like(gamma_correct_img)
        c = 1
        for i in range (256):
            idx = (gamma_correct_img == i)
            new_intnsty = c*(float(i)**gamma)
            new_img_2[idx] = int(new_intnsty)
        gamma_correct_img = new_img_2
        self.__mdfd_img = gamma_correct_img
        self.disp("Gamma transformation")# to display the changed image
        print("Gamma transformation Applied") # Print status to terminal or IDE

    def log_transform(self):
        self.__mdfd_img_lstchg = self.__mdfd_img # store the last changed image data for undo method
        log_trnsfrm_img = self.__mdfd_img
        new_img_3 = np.empty_like(log_trnsfrm_img)
        c = 100
        for i in range (256):
            idx = (log_trnsfrm_img == i)
            new_intnsty = float(c*(math.log10(i+1)))
            new_img_3[idx] = new_intnsty
        log_trnsfrm_img = new_img_3
        self.__mdfd_img = log_trnsfrm_img
        self.disp("Log transformation")# to display the changed image
        print("Log transformation Applied") # Print status to terminal or IDE

    def blur_img_scr_bar(self):
        self.lbl_s3.resize(500,50)
        self.lbl_s3.setText("Please Enter an Integer value Sigma for Gaussian Blur")
        self.lbl_s3.move(100,590)
        self.lbl_s3.show()
        self.e2.setValidator(QIntValidator())
        self.e2.move(500,600)
        btn_blur_img = QPushButton('OK', self)
        btn_blur_img.resize(50,30)
        btn_blur_img.move(610, 600)
        btn_blur_img.show()
        self.e2.show()
        btn_blur_img.clicked.connect(lambda: self.blur_img(int(self.e2.text())))

    def blur_img(self,sigma):
        self.__mdfd_img_lstchg = self.__mdfd_img # store the last changed image data for undo method
        x_count = -1
        y_count = -1
        filter = np.zeros((3,3), dtype=np.float)
        blur_img = self.__mdfd_img
        padd_blur_img = np.insert(blur_img,[0],0,axis = 0)
        padd_blur_img = np.insert(padd_blur_img,[self.__img_height+1],0,axis = 0)
        padd_blur_img = np.insert(padd_blur_img,[0],0,axis = 1)
        padd_blur_img = np.insert(padd_blur_img,[self.__img_width+1],0,axis = 1)
        new_img_4 = np.empty_like(blur_img)

        # print(padd_blur_img[:,0],padd_blur_img[0,:],padd_blur_img[self.__img_height+1,:],padd_blur_img[:,self.__img_width+1])
        for x in range(-1,1):
            x_count+=1
            y_count = -1
            for y in range(-1,1):
                y_count+=1
                filter[x_count,y_count] = math.exp(-(x**2.0+y**2.0)/(2.0*sigma*sigma))
        neighbourhood = np.zeros((3,3))
        progress = 0
        self.dialog.forceShow()
        for j in range(1,self.__img_height+1):
            for k in range(1,self.__img_width+1):
                neighbourhood[0,:] = padd_blur_img[j-1][k-1:k+2]
                neighbourhood[1,:] = padd_blur_img[j-1][k-1:k+2]
                neighbourhood[2,:] = padd_blur_img[j-1][k-1:k+2]
                new_img_4[j-1,k-1] = np.sum(neighbourhood*filter,dtype=np.float)/np.sum(filter,dtype=np.float)
                # print(j ,k)
                progress = progress + 1
                self.dialog.setValue(progress)
                if(self.dialog.wasCanceled()):
                    break
        self.__mdfd_img = new_img_4
        self.disp("Blurred Image",1)# to display the changed image
        print("Image Blurred") # Print status to terminal or IDE

    def sharpen_img_scr_bar(self):
        self.s2.resize(20,400)
        self.s2.move(1330,100)
        self.s2.setMaximum(10)
        self.s2.setMinimum(1)
        self.s2.show()
        self.lbl_s1.resize(30,50)
        self.lbl_s1.setText("High")
        self.lbl_s1.move(1320,500)
        self.lbl_s1.show()
        self.lbl_s2.resize(30,50)
        self.lbl_s2.setText("Low")
        self.lbl_s2.move(1320,50)
        self.lbl_s1.show()
        self.lbl_s2.show()
        self.__mdfd_img_lstchg = self.__mdfd_img # store the last changed image data for undo method
        filter = np.zeros((3,3), dtype=np.float)
        sharp_img = self.__mdfd_img
        padd_sharp_img = np.insert(sharp_img,[0],0,axis = 0)
        padd_sharp_img = np.insert(padd_sharp_img,[self.__img_height+1],0,axis = 0)
        padd_sharp_img = np.insert(padd_sharp_img,[0],0,axis = 1)
        padd_sharp_img = np.insert(padd_sharp_img,[self.__img_width+1],0,axis = 1)
        new_img_5 = np.empty_like(sharp_img)

        # print(padd_blur_img[:,0],padd_blur_img[0,:],padd_blur_img[self.__img_height+1,:],padd_blur_img[:,self.__img_width+1])
        # x_count = -1
        # y_count = -1
        # for x in range(-1,1):
        #     x_count+=1
        #     y_count = -1
        #     for y in range(-1,1):
        #         y_count+=1
        #         filter[x_count,y_count] = -(1.0-(x**2.0+y**2.0)/(2.0*sigma*sigma))*math.exp(-(x**2.0+y**2.0)/(2.0*sigma*sigma))
        filter = [[-1.0,-1.0,-1.0],[-1.0,8.0,-1.0],[-1.0,-1.0,-1.0]]
        # print(len(filter),len(filter[0]))
        neighbourhood = np.zeros((3,3))
        progress = 0
        self.dialog.forceShow()
        for j in range(1,self.__img_height+1):
            for k in range(1,self.__img_width+1):
                progress = progress+1
                neighbourhood[0,:] = padd_sharp_img[j-1][k-1:k+2]
                neighbourhood[1,:] = padd_sharp_img[j-1][k-1:k+2]
                neighbourhood[2,:] = padd_sharp_img[j-1][k-1:k+2]
                # new_img_5[j-1,k-1] = np.sum(neighbourhood*filter,dtype=np.float)/np.sum(filter,dtype=np.float)
                new_img_5[j-1,k-1] = np.sum(neighbourhood*filter,dtype=np.float)
                self.dialog.setValue(progress)
                if(self.dialog.wasCanceled()):
                    break
                # print(j ,k)
        np.clip(new_img_5, 0,255, out=new_img_5)
        self.s2.valueChanged.connect(lambda: self.sharpen_img(new_img_5))

    def sharpen_img(self,new_img_5): # to sharpen the image
        sigma = self.s2.value()
        new_img = np.empty_like(new_img_5)
        np.clip(sigma*new_img_5,0,255,out=new_img)
        self.__mdfd_img = self.__img_v+new_img
        self.disp("Sharpened Image",1)# to display the changed image
        print("Image Sharpened") # Print status to terminal or IDE

    def undoall(self): # to undo all changes done on the image
        self.__mdfd_img = self.__img_v # change the data in the current changed data to original image data
        # self.__mdfd_img_lstchg = self.__img_v
        self.disp("All changes undone") # To Display title
        print("All changes UNDONE ") # Print status to terminal or IDE

    def undo(self): #to undo the last change done on the image
        self.__mdfd_img = self.__mdfd_img_lstchg #update the current changed image as last changed image
        self.disp("Last change undone") # To Display title
        print("Last change UNDONE ")# Print status to terminal or IDE

    def save_image(self): # this method is used for saving the image to the file
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File','','Images (*.png *.xpm *.jpg *.jpeg)')
        itos = cv.merge([self.__img_h,self.__img_s, self.__mdfd_img])
        itos = cv.cvtColor(itos, cv.COLOR_HSV2RGB)
        img_to_save = QtGui.QPixmap(QtGui.QImage(itos,self.__img_width, self.__img_height,3*self.__img_width, QtGui.QImage.Format_RGB888))
        if img_to_save.save(name):
            print("Image Saved To file") # Print status to terminal or IDE
        else:
            print("Could not save the Image to folder") # Print status to terminal or IDE

    def win_close(self): # this method is used for closing the window
        print("Window closed") # Print status to terminal or IDE
        sys.exit() #exit the application

    def disp(self,txt,flag = 0): # this method is used to display the transformed image to GUI
        if (flag == 0):
            self.lbl_s3.clear()
            self.e2.clear()
            self.e2.hide()
            self.lbl_s1.clear()
            self.lbl_s2.clear()
        img_pix1 = cv.merge([self.__img_h,self.__img_s, self.__mdfd_img])
        img_color = cv.cvtColor(img_pix1, cv.COLOR_HSV2RGB)
        pix_img = QtGui.QPixmap(QtGui.QImage(img_color,self.__img_width, self.__img_height,3*self.__img_width, QtGui.QImage.Format_RGB888))
        self.lbl2.clear()
        self.lbl2.setText(txt)
        self.lbl2.resize(200,50)
        self.lbl2.move(950,0)
        self.lbl2.show()
        pix_img= pix_img.scaled(600,600, QtCore.Qt.KeepAspectRatio)
        self.lbl3.clear()
        self.lbl3.resize(600,600)
        self.lbl3.move(720,40)
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
