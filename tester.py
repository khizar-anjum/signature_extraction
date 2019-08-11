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

h = extractor()
h.prepare('pdfs/5_Scan_18022019_192748.pdf')
ocr = h.get_OCR()


#%%
from pdf2image import convert_from_path 
imgs = convert_from_path('pdfs/5_Scan_18022019_192748.pdf', 200)
imgs = [np.array(image,dtype='uint8') for image in imgs]
#img = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
img = imgs[0]
plt.imshow(img,cmap='gray')

#%%
#img_blur = cv2.medianBlur(img, 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
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
rotated = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#s = image_to_string(rotated)
#plt.imshow(rotated,cmap='gray')
#img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#%% 
whatup = exp.copy()
contours, hier = cv2.findContours(whatup,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
prms = []
for i,c in enumerate(contours):
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a white rectangle to visualize the bounding rect
    cv2.rectangle(whatup, (x, y), (x + w, y + h), 255, 1)
    #cv2.imwrite("imgs\\"+str(i)+"output.png",rotated[y:y+h,x:x+w])
    prms.append(cv2.arcLength(c,True))

cv2.drawContours(whatup, contours, -1, (255, 255, 0), 1)
plt.imshow(whatup,cmap='gray')
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
img1 = cv2.bitwise_not(cv2.threshold(cv2.imread('sigs\\A_2.png',0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])# queryImage
img2 = good_imgs[-2] # trainImage

#%%
# Initiate ORB detector
def orb_matcher(img1,img2):
    orb = cv2.ORB_create()
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
    try:
        matches = flann.knnMatch(des1,des2,k=2)
    except:
        return False
    matches = [x for x in matches if len(x) == 2]
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)
    
    if len(good)>MIN_MATCH_COUNT:
        return True
    else: return False
#%%
"""
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


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3)
plt.show()    
"""
#%%
im = img
# resize image
newHeight = 200
newWidth = int(im.shape[1]*200/im.shape[0])
im = cv2.resize(im, (newWidth, newHeight))    
 
# create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
# set input image on which we will run segmentation
ss.setBaseImage(im)
ss.switchToSelectiveSearchQuality()
 
# run selective search segmentation on input image
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))
#%% 
# create a copy of original image
imOut = im.copy()
 
# itereate over all the region proposals
for i, rect in enumerate(rects):
    # draw rectangle for region proposal till numShowRects
    x, y, w, h = rect
    cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

# show output
plt.imshow(imOut)
    
#%%
def get_overlap_area(box_a, box_b):
    x1_a, y1_a, width_a, height_a = box_a
    x1_b, y1_b, width_b, height_b = box_b

    x2_a = x1_a + width_a
    y2_a = y1_a + height_a
    x2_b = x1_b + width_b
    y2_b = y1_b + height_b

    #get the width and height of overlap rectangle
    overlap_width =  min(x2_a, x2_b) - max(x1_a, x1_b) 
    overlap_height = min(y2_a, y2_b) - max(y1_a, y1_b) 

    #If the width or height of overlap rectangle is negative, it implies that two rectangles does not overlap.
    if overlap_width > 0 and overlap_height > 0:
        return overlap_width * overlap_height
    else:
        return 0
  

def get_IOU(box_a, box_b):
    overlap_area = get_overlap_area(box_a, box_b)
    
    #Union = A + B - I(A&B)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = area_a + area_b - overlap_area
    
    
    if overlap_area > 0 :
        return overlap_area / union_area
    else:
        return 0
    
    

#%%
areas = []
for i, rect in enumerate(rects[:-1,...]):
    for j in rects[i+1:,...]:
        temp = get_IOU(rect,j)
        if(temp > 0.80): 
            areas.append(i)
            break
        
#%%
new_rects = np.delete(rects,obj = areas,axis=0)
imOut = im.copy()
 
# itereate over all the region proposals
for i, rect in enumerate(new_rects):
    # draw rectangle for region proposal till numShowRects
    x, y, w, h = rect
    cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

# show output
plt.imshow(imOut)
#%%
frac = img.shape[0]/200
new_rects = new_rects*frac
new_rects = new_rects.astype('int32')
#%%
imOut = img.copy()
 
# itereate over all the region proposals
for i, rect in enumerate(atemp):
    # draw rectangle for region proposal till numShowRects
    x, y, w, h = rect
    cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

# show output
plt.imshow(imOut)
#%%
for i, rect in enumerate(new_rects):
    x, y, w, h = rect
    cv2.imwrite('imgs\\output'+str(i)+'.jpg',img[y:y+h,x:x+w])
    
#%%
# The rects are present in new_rects. Now, I need a scheme for matching images
# and this block does exactly that. 
sig = cv2.imread('sigs\\B_3.png')
matches = []
det_rects = []
for i, rect in enumerate(new_rects):
    x, y, w, h = rect
    if(orb_matcher(sig,img[y:y+h,x:x+w])): 
        matches.append(i)
        det_rects.append(rect)
        
#%%
areas = []
for i, rect in enumerate(det_rects[:-1]):
    temp = []
    for j in det_rects[i+1:]:
        temp.append(get_IOU(rect,j))
    areas.append(np.sum(np.array(temp)))
matches = np.array(matches)[np.argsort(areas)]

#%%
# so two things are nice. one areas and other is matches in ordered form
# a rudimentary policy may be to select 2 top contenders for matches
# i.e. last two elements of matches list with the condition that the
# last two entries of areas list is greater than some threshold.
# right now, 1.5 seems like a good value for that threshold


#%%
from extractor import extractor
ext = extractor()
ext.prepare('pdfs/5_Scan_18022019_192748.pdf')
#%%
ext.load_signature('sigs\\B_3.png','this is second gen')
ext.load_signature('sigs\\A_2.png','this is first gen')
#%%
ext.payload = []
_,_,w1 = ext.extract()