#!/usr/bin/python

import sys # import libraries needed
import PyQt4
import math
import numpy as np
import cv2 as cv
# import PythonQwt as qwt
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
        self.lbl_ker_img = QtGui.QLabel(self) # create a Qlabel object to display kernel
        self.lbl_ker = QtGui.QLabel(self)  #create a Qlabel object to display title for kernel
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
        btn1 = QtGui.QPushButton("Upload Kernel ",self)
        btn1.clicked.connect(self.file_open_kernel) # go to DFT method when clicked on Fing DFT
        btn1.resize(200,40) # resize the button to the required size
        btn1.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        btn1.move(500,100 )
        btn2 = QtGui.QPushButton("Inverse Filter",self)
        btn2.clicked.connect(self.inverse_fliter) # go to inverse_fliter method when clicked on Inverse Filter button
        btn2.resize(200,40) # resize the button to the required size
        btn2.move(500,150 ) # reposition the button at the required position
        btn3 = QtGui.QPushButton("Inverse Filter with inbuilt",self)
        btn3.clicked.connect(self.inv_inbuilt) # go to log_transform method when clicked on Log transform button
        btn3.resize(200,40) # resize the button to the required size
        btn3.move(500,200 ) # reposition the button at the required position
        # btn4 = QtGui.QPushButton("Blur Image",self)
        # btn4.clicked.connect(self.blur_img_scr_bar) # go to blur_img_scr_bar method when clicked on Blur Image button
        # btn4.resize(200,40) # resize the button to the required size
        # btn4.move(500,250 ) # reposition the button at the required position
        # btn5 = QtGui.QPushButton("Sharpening",self)
        # btn5.clicked.connect(self.sharpen_img_scr_bar) # go to sharpen_img_scr_bar method when clicked on Sharpeninge button
        # btn5.resize(200,40) # resize the button to the required size
        # btn5.move(500,300 ) # reposition the button at the required position
        # btn6 = QtGui.QPushButton("Sobel Operator",self)
        # btn6.clicked.connect(self.edge_detect) # go to save_image method when clicked on Sobel operator button
        # btn6.resize(200,40) # resize the button to the required size
        # btn6.move(500,350 ) # reposition the button at the required position
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
        # get image properties.
        self.__img_b,self.__img_g,self.__img_r = cv.split(self.__ip_img)
        self.__img_height,self.__img_width = self.__img_r.shape
        # Image.merge("RGB",(imr,img,imb))
        self.__mdfd_img_lstchg = None
        self.__mdfd_img = None
        self.__mdfd_img_lstchg = self.__img_v # update the last changed value to uploaded image
        self.__mdfd_img = self.__img_v # update the current changed image to uploaded image
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

    def inverse_fliter(self): # method to do inverse filtering
        rw_add = np.ceil((self.__img_height-self.__kernel_height)/2)
        rw_add = rw_add.astype(int)
        col_add = np.ceil((self.__img_width-self.__kernel_width)/2)
        col_add = col_add.astype(int)
        padd_kernel = np.append(np.zeros((rw_add,self.__kernel_width)), self.__kernel, axis=0)#padd with zeros
        padd_kernel = np.append(padd_kernel,np.zeros((rw_add,self.__kernel_width)), axis=0)#padd with zeros
        padd_kernel = np.append(np.zeros((padd_kernel.shape[0],col_add)), padd_kernel,axis=1)#padd with zeros
        padd_kernel = np.append(padd_kernel,np.zeros((padd_kernel.shape[0],col_add)),axis=1)#padd with zeros
        rem_row = self.__img_height-padd_kernel.shape[0]
        rem_col = self.__img_width -padd_kernel.shape[1]
        if(rem_row>0):
            self.__ip_img = np.delete(self.__ip_img, rem_row, 0)
            self.__img_height -= rem_row

        if(rem_col>0):
            self.__ip_img = np.delete(self.__ip_img, rem_col, 1)
            self.__img_width -= rem_col

        H = self.DFT(padd_kernel,1)
        B,G,R = self.DFT(self.__ip_img)
        INV_B = B/H
        INV_G = G/H
        INV_R = R/H

        ib = self.IDFT(INV_B)
        ig = self.IDFT(INV_G)
        ir = self.IDFT(INV_R)
        self.__img_b = (np.absolute(ib)).astype(self.__ip_img.dtype)
        self.__img_g = (np.absolute(ig)).astype(self.__ip_img.dtype)
        self.__img_r = (np.absolute(ir)).astype(self.__ip_img.dtype)
        self.disp("Inverse Filter Applied")

    def inv_inbuilt(self):
        rw_add = np.ceil((self.__img_height-self.__kernel_height)/2)
        rw_add = rw_add.astype(int)
        col_add = np.ceil((self.__img_width-self.__kernel_width)/2)
        col_add = col_add.astype(int)
        padd_kernel = np.append(np.zeros((rw_add,self.__kernel_width)), self.__kernel, axis=0)#padd with zeros
        padd_kernel = np.append(padd_kernel,np.zeros((rw_add,self.__kernel_width)), axis=0)#padd with zeros
        padd_kernel = np.append(np.zeros((padd_kernel.shape[0],col_add)), padd_kernel,axis=1)#padd with zeros
        padd_kernel = np.append(padd_kernel,np.zeros((padd_kernel.shape[0],col_add)),axis=1)#padd with zeros
        rem_row = self.__img_height-padd_kernel.shape[0]
        rem_col = self.__img_width -padd_kernel.shape[1]
        if(rem_row>0):
            print(rem_row)
            self.__ip_img = np.delete(self.__ip_img, rem_row, 0)
            self.__img_height -= rem_row

        if(rem_col>0):
            self.__ip_img = np.delete(self.__ip_img, rem_col, 1)
            self.__img_width -= rem_col
        H = np.fft.fft2(padd_kernel)
        self.__img_b,self.__img_g,self.__img_r = cv.split(self.__ip_img)
        B = np.fft.fft2(self.__img_b)
        G = np.fft.fft2(self.__img_g)
        R = np.fft.fft2(self.__img_r)
        INV_B = B/H
        INV_G = G/H
        INV_R = R/H
        self.__img_b = (np.absolute(np.fft.ifft2(INV_B))).astype(self.__ip_img.dtype)
        self.__img_g = (np.absolute(np.fft.ifft2(INV_G))).astype(self.__ip_img.dtype)
        self.__img_r = (np.absolute(np.fft.ifft2(INV_R))).astype(self.__ip_img.dtype)

        self.disp("kernel transformed")
        print("Inverse Filtering using inbuilt functions ") # Print status to terminal or IDE

    def blur_img_scr_bar(self):
        self.lbl_s3.resize(500,50)#label to display title for output image
        self.lbl_s3.setText("Please Enter an Integer value Sigma for Gaussian Blur")#title text
        self.lbl_s3.move(100,590) #positioning
        self.lbl_s3.show() #display title
        self.e2.setValidator(QIntValidator())#text box setting to allow only integer values
        self.e2.move(500,600) #positioning
        btn_blur_img = QPushButton('OK', self) #button to click ok to start operaion on the input
        btn_blur_img.resize(50,30) #resize the button
        btn_blur_img.move(610, 600) #positioning
        btn_blur_img.show() #display button
        self.e2.show() #display text box
        btn_blur_img.clicked.connect(lambda: self.blur_img(int(self.e2.text()))) #call blur_img when clicked

    def blur_img(self,sigma):
        self.__mdfd_img_lstchg = self.__mdfd_img # store the last changed image data for undo method
        x_count = -1#initialise
        y_count = -1#initialise
        filter = np.zeros((2*sigma+1,2*sigma+1), dtype=np.float) #empty filter kernel
        blur_img = self.__mdfd_img#take the data to temp array
        padd_blur_img = np.append(np.zeros((sigma,self.__img_width)), blur_img, axis=0)#padd with zeros
        padd_blur_img = np.append(padd_blur_img,np.zeros((sigma,self.__img_width)), axis=0)#padd with zeros
        padd_blur_img = np.append(np.zeros((self.__img_height+2*sigma,sigma)), padd_blur_img,axis=1)#padd with zeros
        padd_blur_img = np.append(padd_blur_img,np.zeros((self.__img_height+2*sigma,sigma)),axis=1)#padd with zeros
        new_img_4 = np.empty_like(blur_img)#empty array for storing the output
        for x in range(-sigma,sigma+1):#for the rows of the filter
            x_count+=1
            y_count = -1
            for y in range(-sigma,sigma+1):#for the columns of the filter
                y_count+=1
                filter[x_count,y_count] = math.exp(-(x**2.0+y**2.0)/(2.0*sigma*sigma))#compute the gaussian blur kernel
        neighbourhood = np.zeros((2*sigma+1,2*sigma+1))#window
        progress = 0#to display progress in progres bar
        self.dialog.forceShow() # show the progress bar
        for j in range(sigma,self.__img_height+sigma):#for the rows of the image
            for k in range(sigma,self.__img_width+sigma): #for the columns of the images
                neighbourhood = padd_blur_img[j-sigma:j+sigma+1,k-sigma:k+sigma+1]#take the pixels in neighbourhood of the pixel
                new_img_4[j-sigma,k-sigma] = np.sum(neighbourhood*filter,dtype=np.float)/np.sum(filter,dtype=np.float)#multiply window with filter and average over the filter
                progress = progress + 1#increment the status
                if(progress%5000==0):#display progress every 5000 loops
                    self.dialog.setValue(progress)#to display progress
                if(self.dialog.wasCanceled()):#if the cancel button is pressed
                    break # stop the loog
        self.dialog.setValue(progress) # set the progress
        self.__mdfd_img = new_img_4 #store the computed values in global variable
        self.disp("Blurred Image",1)# to display the changed image
        print("Image Blurred") # Print status to terminal or IDE

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

    def disp(self,txt,flag = 0,scroll = 0,fft=0,img = np.empty_like([256,256])): # this method is used to display the transformed image to GUI
        if (fft == 0): #whether to clear some labels or not is decided by this flag variable
            self.s2.hide() #to hide the scroll bar
            self.lbl_s3.clear() #to clear the label to show new objects
            self.e2.clear() #to clear the label to show new objects
            self.e2.hide() #to hide the text box
            self.lbl_s1.clear() #to clear the label to show new objects
            self.lbl_s2.clear() #to clear the label to show new objects
            if (scroll == 0):#if the button other than sharpen is pressed
                self.s2.setValue(1) #reset the value every time
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
