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
plt.imshow(img,cmap='gray')

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
#plt.imshow(rotated,cmap='gray')
#img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#%% 
#contours, hier = cv2.findContours(rotated, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
prms = []
for i,c in enumerate(contours):
    # get the bounding rect
 #   x, y, w, h = cv2.boundingRect(c)
    # draw a white rectangle to visualize the bounding rect
#    cv2.rectangle(rotated, (x, y), (x + w, y + h), 255, 1)
    #cv2.imwrite("imgs\\"+str(i)+"output.png",rotated[y:y+h,x:x+w])
    prms.append(cv2.arcLength(c,True))

#cv2.drawContours(rotated, contours, -1, (255, 255, 0), 1)

#cv2.imwrite("imgs\\output.png",rotated)

#%%
#img = cv2.imread('fly.png',0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
orb = cv2.ORB_create()
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(rotated, kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(rotated, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()

#%%
img1 = cv2.imread('A_2.png',0)          # queryImage
img2 = cv2.imread('img.png',0) # trainImage
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)
plt.imshow(img3),plt.show()

#%%
# This is for detection of signatures
contours, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
prms = []
for i,c in enumerate(contours):
    prms.append(cv2.arcLength(c,True))
#getting the top 30 contours with high parameters
indices = np.argsort(np.array(prms))[-30:]
good_contours = [contours[i] for i in indices]
good_imgs = []
for c in good_contours:
    x, y, w, h = cv2.boundingRect(c)
    good_imgs.append(rotated[y:y+h,x:x+w])
#%%
img1 = cv2.bitwise_not(cv2.threshold(cv2.imread('A_2.png',0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])# queryImage
img2 = exp.copy()#good_imgs[-3] # trainImage

#%%
# Initiate ORB detector
orb = cv2.BRISK_create()
MIN_MATCH_COUNT = 10
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
        table_number = 6, # 12
        key_size = 12,     # 20
        multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good.append(m)

#%%
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None
    
#%%
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,exp,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()    