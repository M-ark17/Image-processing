import numpy as np
import cv2 as cv

kernel = cv.imread("kernel.png",0)
f = np.fft.fft2(kernel)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
# self.__mdfd_img = magnitude_spectrum # now move the data in temp variable to global variable
cv.imshow("dummy",magnitude_spectrum)
cv.waitKey(0)
