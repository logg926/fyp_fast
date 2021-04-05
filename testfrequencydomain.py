
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import glob
import os
from scipy.interpolate import griddata

# from sklearn.svm import SVC

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def transformFrame(img, size=300):
  f_transform = np.fft.fft2(img)
  # print(f_transform)
  fshift = np.fft.fftshift(f_transform)
  magnitude_spectrum = 20*np.log(np.abs(fshift))
  psd1D = azimuthalAverage(magnitude_spectrum)
  points = np.linspace(0,size,num=psd1D.size)
  xi = np.linspace(0,size,num=size)
  interpolated = griddata(points,psd1D,xi,method='cubic')
  interpolated /= interpolated[0]
  return interpolated

# img = cv2.imread('aamjfukxwp_0.jpg', 0)

# dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
# print(np.float32(img))
# f_transform = np.fft.fft2(img)
# # print(f_transform)
# # 0 = cv2.IMREAD_GRAYSCALE
# # print (img)

# # with open('imgtestfrequencydomain.json', 'w') as f:
# #     json.dump({ "detectimg": img.tolist()}, f)
# size = 300
# feature = transformFrame(img, size)

# import pickle
# SVM = pickle.load(open('./SVM model_v0.24.1.pkl', 'rb'))
# print(SVM)
# print(SVM.predict(np.array([feature])))