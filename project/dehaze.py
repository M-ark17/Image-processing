import cv2 as cv;
import math;
import numpy as np;

def DCP(im,sz):
    b,g,r = cv.split(im) #split image into r,g,b
    dc = cv.min(cv.min(r,g),b); #dark channel will be min of r,g,b
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(sz,sz)) #do this for all patches
    dcp = cv.erode(dc,kernel)
    # print(dcp.shape)
    return dcp

def Atm_est(im,dark):
    [h,w] = im.shape[:2]
    img_sz = h*w
    num_pixels = int(max(math.floor(img_sz/1000),1)) # 0.1% of the pixels
    darkvec = dark.reshape(img_sz,1); #rearrange dark channel into a row
    imvec = im.reshape(img_sz,3); #get the image array into a row

    index = darkvec.argsort(); #sort and get the indices using argsort
    index = index[img_sz-num_pixels::] #take 0.1% of these

    atmsum = np.zeros([1,3])
    for ind in range(1,num_pixels):
       atmsum = atmsum + imvec[index[ind]]  #take sum of these pixels

    A = atmsum / num_pixels; # get the average which will be Atmosphere
    return A

def Tx_est(im,A,sz):
    beta = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]
# once we know the darkchannel transmission is just subtraction form 1
    tx = 1 - beta*DCP(im3,sz);
    return tx

def filtering(im,p,r,eps):
    mean_I = cv.boxFilter(im,cv.CV_64F,(r,r));
    mean_p = cv.boxFilter(p, cv.CV_64F,(r,r));
    mean_Ip = cv.boxFilter(im*p,cv.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv.boxFilter(im*im,cv.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv.boxFilter(a,cv.CV_64F,(r,r));
    mean_b = cv.boxFilter(b,cv.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def Tx_smoothing(im,et):
    gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = filtering(gray,et,r,eps); # use guided filter to remove blocking artifacts

    return t;

def Dehaze_img(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind] #we have A and t get J from I

    return res

if __name__ == '__main__':
    import sys
    import os
    cwd = os.getcwd()
    if (len(sys.argv)==1):
        files = os.listdir(cwd+"/haze_img")
        print ('files in the folder "clear_img" are ',files)
        flag = 1
    else:
        files = os.listdir(sys.argv[1])
        print ('files in the folder "'+sys.argv[1]+'" are ',files)
        flag = 0
    if (os.path.isdir(cwd+"/clear_img")):
        pass
    else:
        try:
            os.makedirs(cwd+"/clear_img")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if (os.path.isdir(cwd+"/DCP")):
        pass
    else:
        try:
            os.makedirs(cwd+"/DCP")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if (os.path.isdir(cwd+"/transmission")):
        pass
    else:
        try:
            os.makedirs(cwd+"/transmission")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    for i in files:
        if i.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = cv.imread(cwd+"/haze_img/"+i);

            I = src.astype('float64')/255.0;

            dcp = DCP(I,15);
            A = Atm_est(I,dcp);
            te = Tx_est(I,A,15);
            t = Tx_smoothing(src,te);
            J = Dehaze_img(I,t,A,0.1);

            cv.imwrite(cwd+'/DCP/' + "dcp_"+i,dcp*255.0)
            cv.imwrite(cwd+'/transmission/' + "t_"+i,t*255.0);
            cv.imwrite(cwd+'/clear_img/' + "J_"+i,J*255.0);
            # cv.imwrite("./image/J.png",J*255);
            # cv.waitKey();
