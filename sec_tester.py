# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:51:01 2019

@author: khizar
"""
img1 = cv2.bitwise_not(cv2.threshold(cv2.imread('A_2.png',0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])# queryImage
img2 = good_imgs[-3] # trainImage

#%%
# Initiate ORB detector
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
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

#%%\
img = rotated.copy()

black = np.zeros_like(rotated)
#for cnt in good_contours:
cnt = good_contours[-3]
hull = cv2.convexHull(cnt)

img3 = img2.copy()
black2 = black.copy()

#--- Here is where I am filling the contour after finding the convex hull ---
cv2.drawContours(black2, [hull], -1, (255, 255, 255), -1)
plt.imshow(black2)

masked = cv2.bitwise_and(img, img, mask = black2)    
x, y, w, h = cv2.boundingRect(cnt)
good_sig = masked[y:y+h,x:x+w]
plt.imshow(good_sig,cmap='gray')

#%%
#kernel = np.ones((3,3),np.uint8)
exp = rotated.copy() #cv2.dilate(rotated,kernel) #
n = 13
linek = np.zeros((n,n),dtype=np.uint8)
linek[int((n-1)/2),...]=1
x=cv2.morphologyEx(rotated, cv2.MORPH_OPEN, linek ,iterations=5)
exp-=x
linek = np.zeros((n,n),dtype=np.uint8)
linek[...,int((n-1)/2)]=1
x=cv2.morphologyEx(exp, cv2.MORPH_OPEN, linek ,iterations=5)
exp-=x
plt.imshow(exp,cmap='gray')

#%%
for i,im in enumerate(good_imgs):
    h, w = im.shape[:2]
    #cv2.imwrite("imgs\\"+str(i)+"output.png",im)
    print(i, w/h)
#should be greater than 0.9
