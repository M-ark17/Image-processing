import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
kernel = cv.imread("kernel/kernel.png",0)
kernel = kernel.astype(np.float)
img = cv.imread("ground_truth/GroundTruth1_1_1.jpg",0)
kernel = np.pad(kernel,[(0,800-21),(0,800-21)],"constant")/np.sum(kernel)
plt.imshow(kernel)
f = np.fft.fft2(kernel)
img_f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
img_sh = np.fft.fftshift(img_f)
# np.fft.ifft2(fshift*img_sh)

# magnitude_spectrum = 20*np.log(np.abs(fshift))
# self.__mdfd_img = magnitude_spectrum # now move the data in temp variable to global variable
# plt.imshow(np.real(np.fft.ifftshift(np.fft.ifft2(fshift*img_sh))))
# cv.waitKey(0)
plt.show()
