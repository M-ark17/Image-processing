# 08-May-2015, Behzad Tabibian

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def mv2pm(mv, K, mm2pixel):
  RY1 = np.array([[np.cos(mv[0]), 0, np.sin(mv[0])],
                  [0, 1, 0],
                  [-np.sin(mv[0]),0, np.cos(mv[0])]])
  RX2 = np.array([[1.0,0.0,0.0],
                  [0,np.cos(mv[1]), -np.sin(mv[1])],
                  [0, np.sin(mv[1]), np.cos(mv[1])]])
  RZ3 = np.array([[np.cos(mv[2]), -np.sin(mv[2]), 0],
                  [np.sin(mv[2]), np.cos(mv[2]), 0],
                  [0, 0, 1]])
  R = RX2.dot(RY1).dot(RZ3)
  t = mv[3:]*mm2pixel
  P = K.dot(np.vstack((R.T, t)).T)
  return P

def PSFprojection(walk,params):
  cols, rows = params['RowColPixels']
  chip_x, chip_y = params['SensorSize']
  f = params['FocalLength']
  D = params['Distance']
  crop = params['Crop']
  mm2pixel = rows/chip_y
  f_px = f*cols/chip_x
  Kext = np.array([[f_px,0,crop/2],[0,f_px,crop/2],[0,0,1.0]])
  x = np.arange(cols/2-350,cols/2+351,  50 )-cols/2;
  y = np.arange(rows/2-350, rows/2+351,50)-rows/2;
  X, Y = np.meshgrid(x,y)
  M = np.vstack((X.T.flatten()/f*D,Y.T.flatten()/f*D,
                 np.ones((1, X.size))*D*mm2pixel,np.ones((1, X.size))))
  ker = np.zeros((walk.shape[2],crop,crop))
  for iImg in xrange(walk.shape[2]):
    motion = iImg
    kernel = np.zeros((crop,crop))
    mv = np.zeros(6)
    for i in xrange(walk.shape[0]):
      walkCur = walk[i,:,motion]
      mv[0] =  walkCur[2] * np.pi / 180.0
      mv[1] =- walkCur[0] * np.pi / 180.0
      mv[2] =- walkCur[1] * np.pi / 180.0
      mv[3] =- walkCur[3]
      mv[4] =  walkCur[5]
      mv[5] =- walkCur[4]
      P = mv2pm(mv, Kext,mm2pixel)
      m = P.dot(M);
      x = m[0, :] / m[2, :];
      y = m[1, :] / m[2, :];
      L =  (np.round(x)<1) | (np.round(x)>crop) | (np.round(y)<1) | (np.round(y)>crop)
      I = np.logical_not(L)
      kernel[np.round(x[I]).astype(int)-1,np.round(y[I]).astype(int)-1] = kernel[np.round(x[I]).astype(int)-1,np.round(y[I]).astype(int)-1] + 1;
    ker[iImg,:,:] = kernel/walk.shape[0]
  return ker

param = {'RowColPixels':(1872.0,2808.0),
         'SensorSize': (24.0,36.0),
         'FocalLength': 50.0,
         'Distance': 620.0,
         'Crop': 800.0
         }
walk = scipy.io.loadmat("./500Frames/walkAll.mat")
ker = PSFprojection(walk['walk'],param)

def plot(i=[0,ker.shape[0]-1]):
  plt.figure(figsize=(14,14))
  plt.imshow(ker[i,:].T*5.0,interpolation='nearest',cmap=cm.Greys_r)
