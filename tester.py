# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:10:11 2019

@author: khizar
"""

from extractor import extractor
import cv2
import numpy as np
from pytesseract import image_to_string
from matplotlib import pyplot as plt

h = extractor('what is up')
h.prepare('pdfs/2_Scan_18022019_192700.pdf')
ocr = h.get_OCR()


#%%
from pdf2image import convert_from_path 
imgs = convert_from_path('pdfs/2_Scan_18022019_192700.pdf', 200)
imgs = [np.array(image,dtype='uint8') for image in imgs]
img = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY) 


#%%
#img_blur = cv2.medianBlur(img, 3)
img_thresh_Gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
gray = cv2.bitwise_not(img_thresh_Gaussian)
# Rotation code
coords = np.column_stack(np.where(gray > 0))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle

(h, w) = gray.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(gray, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

s = image_to_string(rotated)
plt.imshow(rotated,cmap='gray')
#img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#%% 
contours, hier = cv2.findContours(rotated, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a white rectangle to visualize the bounding rect
    cv2.rectangle(rotated, (x, y), (x + w, y + h), 255, 1)

cv2.drawContours(rotated, contours, -1, (255, 255, 0), 1)

cv2.imwrite("output.png",rotated)

#%%
from signature_extractor import signature_extractor