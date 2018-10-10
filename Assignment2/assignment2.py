#!/usr/bin/python

import sys # import libraries needed
import PyQt4
import math
import numpy as np
import cv2 as cv
from scipy.signal import convolve2d
# import PythonQwt as qwt
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4.QtCore import pyqtSlot,SIGNAL,SLOT

class Window(QtGui.QMainWindow): #create a class to display a window

    def __init__(self): #method to declare attributes of the class
        super(Window,self).__init__()
        self.setGeometry(50,50,1400,650)    # to set the size of the window to 1400*650
        self.setWindowTitle("Image Restoration Tool") # give title to the window
        self.home() #method called home will have all the main features of the GUI
        self.__pixmap = None # create pixmap attribute to display image to GUI
        self.__img_height = None # height of the Image
        self.__img_width = None # widht of the Image
        self.lbl = QtGui.QLabel(self)  # create a Qlabel object to display input image
        self.lbl1 = QtGui.QLabel(self) # create a Qlabel object to display title for input image
        self.lbl_ker_img = QtGui.QLabel(self) # create a Qlabel object to display kernel
        self.lbl_ker = QtGui.QLabel(self)  #create a Qlabel object to display title for kernel
        self.lbl2 = QtGui.QLabel(self) # create a Qlabel object to display title for output image
        self.lbl3 = QtGui.QLabel(self) # create a Qlabel object to displat output image
        self.lbl_s3 = QtGui.QLabel(self) # create a Qlabel object to display text for text editor
        self.lbl_s4 = QtGui.QLabel(self) # create a Qlabel object to display text for text editor
        self.lbl_s5 = QtGui.QLabel(self) # create a Qlabel object to display text for text editor
        self.e2 = QtGui.QLineEdit(self) # create a QLineEdit object to display scroll title
        self.e3 = QtGui.QLineEdit(self) # create a QLineEdit object to display scroll title
        self.e4 = QtGui.QLineEdit(self) # create a QLineEdit object to display scroll title


    def home(self): # home method of the QMainWindow
        btn = QtGui.QPushButton("Upload Image",self) # button for uploading image
        btn.clicked.connect(self.file_open) # go to file_open method when clicked on Upload Image button
        btn.resize(200,40) # resize the button to the required size
        btn.move(500,50 ) # reposition the button at the required position
        btn1 = QtGui.QPushButton("Upload Kernel ",self)
        btn1.clicked.connect(self.file_open_kernel) # go to DFT method when clicked on Fing DFT
        btn1.resize(200,40) # resize the button to the required size
        btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        btn1.move(500,100 )
        btn2 = QtGui.QPushButton("Inverse Filter",self)
        btn2.clicked.connect(lambda: self.inverse_fliter(-1)) # go to inverse_fliter method when clicked on Inverse Filter button
        btn2.resize(200,40) # resize the button to the required size
        btn2.move(500,150 ) # reposition the button at the required position
        btn3 = QtGui.QPushButton("Get blur image",self)
        btn3.clicked.connect(self.inv_inbuilt) # go to log_transform method when clicked on Log transform button
        btn3.resize(200,40) # resize the button to the required size
        btn3.move(500,200 ) # reposition the button at the required position
        btn4 = QtGui.QPushButton("Radial Filtering",self)
        btn4.clicked.connect(self.radial_filter_threshold) # go to blur_img_scr_bar method when clicked on Blur Image button
        btn4.resize(200,40) # resize the button to the required size
        btn4.move(500,250 ) # reposition the button at the required position
        btn5 = QtGui.QPushButton("Weiner Filtering",self)
        btn5.clicked.connect(self.weiner_filtering) # go to weiner_filtering method when clicked on Sharpeninge button
        btn5.resize(200,40) # resize the button to the required size
        btn5.move(500,300 ) # reposition the button at the required position
        # btn6 = QtGui.QPushButton("Sobel Operator",self)
        # btn6.clicked.connect(self.edge_detect) # go to save_image method when clicked on Sobel operator button
        # btn6.resize(200,40) # resize the button to the required size
        # btn6.move(500,350 ) # reposition the button at the required position
        btn7 = QtGui.QPushButton("LS Filtering",self)
        btn7.clicked.connect(self.ls_filtering_gamma) # go to undo method when clicked on Undo last Change button
        btn7.resize(200,40) # resize the button to the required size
        btn7.move(500,400 ) # reposition the button at the required position
        btn8 = QtGui.QPushButton("Calculate Metrics ",self)
        btn8.clicked.connect(self.metrics) # go to undoall method when clicked on Undo All Changes button
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
        # get image properties.
        self.__img_b,self.__img_g,self.__img_r = cv.split(self.__ip_img)
        self.__img_height,self.__img_width = self.__img_r.shape
        # Image.merge("RGB",(imr,img,imb))
        self.__mdfd_img_lstchg = None
        self.__mdfd_img = None
        if upld_img.load(name): # if the image is uploaded properly then upld_img.load will be true
            self.lbl1.clear() # clear the past content in label if any is present
            self.lbl1.setText("Orignal Image") # Set title for the input image to display
            self.lbl1.move(200,140) # position the title
            self.lbl1.show() # show the title
            pixmap = QtGui.QPixmap(upld_img) #convert the opencv image to pixmap to display it on GUI
            self.__pixmap = pixmap.scaled(400, 650, QtCore.Qt.KeepAspectRatio) # scale the pixmap to display it on GUI keep the Aspect Ratio of the original image
            self.lbl.clear() # clear the past content in label if any is present
            self.lbl.resize(400,650) # set the size of the input pixmap to 400*650
            self.lbl.move(50,50) # position the input pixmap
            self.lbl.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
            self.lbl.setScaledContents(False)
            self.lbl.setPixmap(self.__pixmap) # set the pixmap to the label
            self.lbl.show()# show the pixmap as image
            print("Selected Image uploaded") #print status to the terminal or IDE
        else: #if the image is not uploaded then
            print("Could not upload Image") # print status to the terminal or IDE

    def file_open_kernel(self): #method to open file
        name = QtGui.QFileDialog.getOpenFileName(self,'Open File','','Images (*.png *.xpm *.jpg *.jpeg)') #this will open a dialog box to upload image only png,xpm,jpg,jpeg images are supported
        upld_img = QtGui.QImage() # create Qimage object to store the uploaded image data
        self.__kernel =  cv.imread(str(name),cv.IMREAD_GRAYSCALE) # upload the image from the dialog box using imread in opencv library
        # get image properties.
        self.__kernel_height,self.__kernel_width = self.__kernel.shape
        # Image.merge("RGB",(imr,img,imb))
        if upld_img.load(name): # if the image is uploaded properly then upld_img.load will be true
            self.lbl_ker.clear() # clear the past content in label if any is present
            self.lbl_ker.setText("kernel") # Set title for the input image to display
            self.lbl_ker.move(225,10) # position the title
            self.lbl_ker.show() # show the title
            pixmap = QtGui.QPixmap(upld_img) #convert the opencv image to pixmap to display it on GUI
            self.__pixmap = pixmap.scaled(100, 125, QtCore.Qt.KeepAspectRatio) # scale the pixmap to display it on GUI keep the Aspect Ratio of the original image
            self.lbl_ker_img.clear() # clear the past content in label if any is present
            self.lbl_ker_img.resize(100,125) # set the size of the input pixmap to 100*125
            self.lbl_ker_img.move(200,25) # position the input pixmap
            self.lbl_ker_img.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
            self.lbl_ker_img.setScaledContents(False)
            self.lbl_ker_img.setPixmap(self.__pixmap) # set the pixmap to the label
            self.lbl_ker_img.show()# show the pixmap as image
            print("Selected Kernel uploaded") #print status to the terminal or IDE
        else: #if the image is not uploaded then
            print("Could not upload kernel") # print status to the terminal or IDE

    def FFT_matrix(self,N,sign=1): #function to compute FFT matrix
        i, j = np.meshgrid(np.arange(N), np.arange(N))
        omega = np.exp( sign * -2 * np.pi * 1J / N )
        W = np.power( omega, i * j ) / np.sqrt(N)
        return W

    def DFT(self,img,ker=0):# this method performs the Discreet fourier Transform
        self.__mdfd_img_lstchg = self.__mdfd_img # store the last changed image data for undo method
        if(ker == 1):
            rows = self.FFT_matrix(img.shape[0])
            cols = self.FFT_matrix(img.shape[1])
            img = rows.dot(img).dot(cols)
            img = np.fft.fftshift(img)
            # cv.imwrite("DFT.jpg",np.absolute(img))
            return img
        else:
            b,g,r = cv.split(img)
            rows = self.FFT_matrix(self.__img_height)
            cols = self.FFT_matrix(self.__img_width)
            b = rows.dot(b).dot(cols)
            b = np.fft.fftshift(b)
            g = rows.dot(g).dot(cols)
            g = np.fft.fftshift(g)
            r = rows.dot(r).dot(cols)
            r = np.fft.fftshift(r)
            # cv.imwrite("DFT.jpg",img)
            return b,g,r

    def IDFT(self,img,ker=0):# this method performs the Inverse Discreet fourier Transform
        rows = self.FFT_matrix(img.shape[0],-1)
        cols = self.FFT_matrix(img.shape[1],-1)
        img = rows.dot(img).dot(cols)
        img = np.fft.ifftshift(img)
        # print("DFT calculated",np.ceil(np.absolute(img))) # Print status to terminal or IDE
        # cv.imwrite("IDFT.jpg",np.absolute(img))
        return img

    def padder(self,img):
        rw_add = np.ceil((self.__img_height-img.shape[0])/2)
        rw_add = rw_add.astype(int)
        col_add = np.ceil((self.__img_width-img.shape[1])/2)
        col_add = col_add.astype(int)
        padd_img = np.append(np.zeros((rw_add,img.shape[1])), img, axis=0)#padd with zeros
        padd_img = np.append(padd_img,np.zeros((rw_add,padd_img.shape[1])), axis=0)#padd with zeros
        padd_img = np.append(np.zeros((padd_img.shape[0],col_add)), padd_img,axis=1)#padd with zeros
        padd_img = np.append(padd_img,np.zeros((padd_img.shape[0],col_add)),axis=1)#padd with zeros
        rem_row = self.__img_height-padd_img.shape[0]
        rem_col = self.__img_width -padd_img.shape[1]
        if(rem_row>0):
            self.__ip_img = np.delete(self.__ip_img, rem_row, 0)
            self.__img_height -= rem_row

        if(rem_col>0):
            self.__ip_img = np.delete(self.__ip_img, rem_col, 1)
            self.__img_width -= rem_col
        return padd_img

    def inverse_fliter(self,sigma = -1): # method to do inverse filtering
        padd_kernel = self.padder(self.__kernel)
        H = self.DFT(padd_kernel,1)
        if(sigma != -1):
            for index, x in np.ndenumerate(H):
                if (np.sqrt(index[0]*index[0]+index[1]*index[1])>sigma):
                    H[index[0],index[0]] = 1
            print("Radial")
        B,G,R = self.DFT(self.__ip_img/256)
        INV_B = B/H
        INV_G = G/H
        INV_R = R/H
        ib = self.IDFT(INV_B)*256
        ig = self.IDFT(INV_G)*256
        ir = self.IDFT(INV_R)*256
        self.__img_b = (np.absolute(ib)).astype(self.__ip_img.dtype)
        self.__img_g = (np.absolute(ig)).astype(self.__ip_img.dtype)
        self.__img_r = (np.absolute(ir)).astype(self.__ip_img.dtype)
        self.disp("Inverse Filter Applied")
        print("Inverse Filter Applied")

    def inv_inbuilt(self):
        motion_blr = cv.filter2D(self.__ip_img,-1,np.divide(self.__kernel,np.sum(self.__kernel).astype(self.__ip_img.dtype)))
        self.__img_b,self.__img_g,self.__img_r = cv.split(motion_blr)
        cv.imwrite("motion_blr.jpg",motion_blr)
        self.disp("Blurred Image")
        print("Blurring using kernel") # Print status to terminal or IDE

    def radial_filter_threshold(self):
        self.lbl_s3.resize(500,50)#label to display title for output image
        self.lbl_s3.setText("Please Enter an Integer value Threshold")#title text
        self.lbl_s3.move(100,590) #positioning
        self.lbl_s3.show() #display title
        self.e2.setValidator(QIntValidator())#text box setting to allow only integer values
        self.e2.move(500,600) #positioning
        radial_threshold = QPushButton('OK', self) #button to click ok to start operaion on the input
        radial_threshold.resize(50,30) #resize the button
        radial_threshold.move(610, 600) #positioning
        self.lbl_s5.clear()
        self.lbl_s4.clear()
        self.e4.clear()
        self.e3.clear()
        radial_threshold.show() #display button
        self.e2.show() #display text box
        radial_threshold.clicked.connect(lambda: self.inverse_fliter(int(self.e2.text()))) #call blur_img when clicked

    def weiner_filtering(self):
        self.lbl_s4.resize(500,50)#label to display title for output image
        self.lbl_s4.setText("Please Enter an Integer value of K")#title text
        self.lbl_s4.move(100,590) #positioning
        self.lbl_s4.show() #display title
        self.e3.setValidator(QIntValidator())#text box setting to allow only integer values
        self.e3.move(500,600) #positioning
        weiner_k = QPushButton('OK', self) #button to click ok to start operation on the input
        weiner_k.resize(50,30) #resize the button
        weiner_k.move(610, 600) #positioning
        weiner_k.show() #display button
        self.lbl_s5.clear()
        self.lbl_s3.clear()
        self.e4.clear()
        self.e2.clear()
        self.e3.show() #display text box
        weiner_k.clicked.connect(lambda: self.weiner(int(self.e3.text()))) #call blur_img when clicked

    def weiner(self,k):
        padd_kernel = self.padder(self.__kernel)
        H = self.DFT(padd_kernel,1)
        B,G,R = self.DFT(self.__ip_img)
        INV_B = np.multiply(B,np.divide(np.power(np.absolute(H),2),(np.multiply(H,np.power(np.absolute(H),2)+k))))
        INV_G = np.multiply(G,np.divide(np.power(np.absolute(H),2),(np.multiply(H,np.power(np.absolute(H),2)+k))))
        INV_R = np.multiply(R,np.divide(np.power(np.absolute(H),2),(np.multiply(H,np.power(np.absolute(H),2)+k))))

        ib = self.IDFT(INV_B)
        ig = self.IDFT(INV_G)
        ir = self.IDFT(INV_R)
        self.__img_b = (np.absolute(ib)).astype(self.__ip_img.dtype)
        self.__img_g = (np.absolute(ig)).astype(self.__ip_img.dtype)
        self.__img_r = (np.absolute(ir)).astype(self.__ip_img.dtype)
        self.disp("Weiner Filter Applied")
        print("Weiner Filter Applied")

    def ls_filtering_gamma(self): # to undo all changes done on the image
        self.lbl_s5.resize(500,50)#label to display title for output image
        self.lbl_s5.setText("Please Enter an Integer value of gamma")#title text
        self.lbl_s5.move(100,590) #positioning
        self.lbl_s5.show() #display title
        self.e4.setValidator(QIntValidator())#text box setting to allow only integer values
        self.e4.move(500,600) #positioning
        gamma = QPushButton('OK', self) #button to click ok to start operaion on the input
        gamma.resize(50,30) #resize the button
        gamma.move(610, 600) #positioning
        gamma.show() #display button
        self.lbl_s3.clear()
        self.lbl_s4.clear()
        self.e2.clear()
        self.e3.clear()
        self.e4.show() #display text box
        gamma.clicked.connect(lambda: self.ls_filter(int(self.e4.text()))) #call blur_img when clicked

        print("LS filtering DONE ") # Print status to terminal or IDE
    def ls_filter(self,gamma=1):
        p = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        padd_p = self.padder(p)
        P = self.DFT(padd_p,1)
        h = self.padder(self.__kernel)
        H = self.DFT(h,1)
        B,G,R = self.DFT(self.__ip_img)
        filter = np.divide(np.conj(H),(np.power(np.absolute(H),2)+gamma*np.power(np.absolute(P),2)))
        R_trans = np.multiply(filter,R)
        G_trans = np.multiply(filter,G)
        B_trans = np.multiply(filter,B)
        ib = self.IDFT(B_trans)
        ig = self.IDFT(G_trans)
        ir = self.IDFT(R_trans)
        self.__img_b = (np.absolute(ib)).astype(self.__ip_img.dtype)
        self.__img_g = (np.absolute(ig)).astype(self.__ip_img.dtype)
        self.__img_r = (np.absolute(ir)).astype(self.__ip_img.dtype)
        self.disp("Weiner Filter Applied")
        print("Weiner Filter Applied")

    def metrics(self): #to undo the last change done on the image

        print("Last change UNDONE ")# Print status to terminal or IDE

    def save_image(self): # this method is used for saving the image to the file
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File','','Images (*.png *.xpm *.jpg *.jpeg)') # tp open a dialog box to input image
        itos = cv.merge([self.__img_b,self.__img_g, self.__img_r])#merge intensity with the hue and saturation
        itos = cv.cvtColor(itos, cv.COLOR_BGR2RGB)#convert hsv to rgb image
        img_to_save = QtGui.QPixmap(QtGui.QImage(itos,self.__img_width, self.__img_height,3*self.__img_width, QtGui.QImage.Format_RGB888)) # convert opencv image to pixmap to display in gui
        if img_to_save.save(name):#if the image is saved
            print("Image Saved To file") # Print status to terminal or IDE
        else:#if the could not be saved
            print("Could not save the Image to folder") # Print status to terminal or IDE

    def win_close(self): # this method is used for closing the window
        print("Window closed") # Print status to terminal or IDE
        sys.exit() #exit the application

    def disp(self,txt,flag = 0,fft=0,img = np.empty_like([256,256])): # this method is used to display the transformed image to GUI
        if (fft == 0): #whether to clear some labels or not is decided by this flag variable
            self.lbl_s3.clear() #to clear the label to show new objects
            self.e2.clear() #to clear the label to show new objects
            self.e2.hide() #to hide the text box
            if (flag == 0 ):
                img_pix1 = cv.merge((self.__img_b,self.__img_g, self.__img_r)) #merge the v with h and s using cv.merge
                # cv.imwrite('Blue Channel.jpg',self.__img_b)
                # cv.imwrite('Green Channel.jpg',self.__img_g)
                # cv.imwrite('Red Channel.jpg',self.__img_r)
                # cv.imwrite('Merged Output.jpg',img_pix1)
                # img_color = cv.cvtColor(img_pix1, cv.COLOR_HSV2RGB) #convert the image to color image
                # img_pix1 = np.dstack((self.__img_b,self.__img_g, self.__img_r))
                img_pix1 = cv.cvtColor(img_pix1, cv.COLOR_BGR2RGB)
                pix_img = QtGui.QPixmap(QtGui.QImage(img_pix1,self.__img_width, self.__img_height,3*self.__img_width, QtGui.QImage.Format_RGB888)) # convert opencv image to pixmap to display it to the user
        else:
            pix_img = QtGui.QPixmap(QtGui.QImage(img,self.__img_width, self.__img_height,3*self.__img_width,QtGui.QImage.Format_Indexed8))
        self.lbl2.clear() #to clear the label to show new objects
        self.lbl2.setText(txt) #set the text to display
        self.lbl2.resize(300,50) #resize the label to required size
        self.lbl2.move(950,0) #positioning the label
        self.lbl2.show() #show the label
        pix_img = pix_img.scaled(600,600, QtCore.Qt.KeepAspectRatio)
        self.lbl3.clear() #to clear the label to show new objects
        self.lbl3.resize(600,600) #resize the label to required size
        self.lbl3.move(720,40) #positioning the label
        self.lbl3.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.lbl3.setScaledContents(False) #keep the image as it is while scaling
        self.lbl3.setPixmap(pix_img) #shoe the image
        self.lbl3.show() #show the label

def main(): # define  a main class to call window created
    app = QtGui.QApplication(sys.argv) # start a qtgui application
    GUI = Window() #create an object of the window
    # GUI.disp() # display it
    sys.exit(app.exec_()) #close the window
main()
