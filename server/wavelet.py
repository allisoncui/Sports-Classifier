import numpy as np
import pywt
import cv2

def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)   # convert to grayscale
    imArray = np.float32(imArray)                         # convert to float
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)    # compute coefficients
    
    coeffs_H = list(coeffs)      # process coeffs
    coeffs_H[0] *= 0
    
    imArray_H = pywt.waverec2(coeffs_H, mode)  # reconstruction
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    
    return imArray_H