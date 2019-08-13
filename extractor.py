# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 21:51:00 2019

@author: khizar
"""

import os
from pdf2image import convert_from_path 
from pytesseract import image_to_string
from skimage import measure
from skimage.measure import regionprops
import imagehash
from PIL import Image
import cv2
import numpy as np

class extractor:
    def __init__(self):
        # name of the current file under operation
        self.current_file = ''
        # list of the images a pdf file is converted to after preperation
        self.images = []
        self.signatures = []
        self.payload = []
        self.sig_hashes = []
        
    def __exists(self,filename):
        if(os.path.exists(filename)):
            return True
        else:
            raise NameError('\''+filename+'\' does not exist in path.')
        
    def __match(self,img1,img2):
        orb = cv2.ORB_create()
        MIN_MATCH_COUNT = 10
        RATIO_FOR_TEST = 0.7
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
            if m.distance < RATIO_FOR_TEST*n.distance:
                good.append(m)
        
        if len(good)>MIN_MATCH_COUNT:
            return True
        else: return False
            
    def __preprocess(self):
        for i,image in enumerate(self.images):
            image = np.array(image,dtype='uint8')
            """
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV\
                                + cv2.THRESH_OTSU)[1]
            # Rotation code
            coords = np.column_stack(np.where(img > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
            	angle = -(90 + angle)
            else:
            	angle = -angle
            
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(image, M, (w, h)\
                ,flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            """
            self.images[i] = self.__get_connected_components(image)
            
    def __get_overlap_area(self,box_a, box_b):
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
      
    
    def __get_IOU(self,box_a, box_b):
        overlap_area = self.__get_overlap_area(box_a, box_b)
        
        #Union = A + B - I(A&B)
        area_a = box_a[2] * box_a[3]
        area_b = box_b[2] * box_b[3]
        union_area = area_a + area_b - overlap_area
        
        
        if overlap_area > 0 :
            return overlap_area / union_area
        else:
            return 0
        
    def __get_selective_matches(self,img):
        #parameters
        MAX_OVERLAP = 0.80
        NEW_HEIGHT = 200
        # resize image
        frac = img.shape[0]/NEW_HEIGHT #the resize factor
        newWidth = int(img.shape[1]/frac)
        img = cv2.resize(img, (newWidth, NEW_HEIGHT))
         
        # create Selective Search Segmentation Object using default parameters
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
         
        # set input image on which we will run segmentation
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchQuality()
         
        # run selective search segmentation on input image
        rects = ss.process()
        print(len(rects))
        #filter upon the basis of maximum overlap allowed
        areas = []            
        for i, rect in enumerate(rects[:-1,...]):
            for j in rects[i+1:,...]:
                temp = self.__get_IOU(rect,j)
                _, _, w, h = rect
                if(temp > MAX_OVERLAP or h/w > 3): 
                    areas.append(i)
                    break
        rects = np.delete(rects,obj=areas,axis=0)
        rects = rects*frac #resizing the reigons again
        rects = rects.astype('int32')
        return rects
    
    def __process_line(self,thresh,output):	
        # assign a rectangle kernel size	1 vertical and the other will be horizontal
        kernel = np.ones((1,5), np.uint8)
        kernel2 = np.ones((2,4), np.uint8)	
        # use closing morph operation but fewer iterations than the letter then erode to narrow the image	
        temp_img = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel2,iterations=2)
        #temp_img = cv2.erode(thresh,kernel,iterations=2)	
        line_img = cv2.dilate(temp_img,kernel,iterations=5)

        (contours, _) = cv2.findContours(line_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        aspects = []
        areas = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspects.append(w/h)
            areas.append(w*h)
            if(w/h > 8 or (10000 < w*h and w*h < 20000) or w*h >200000):
                output[y:y+h,x:x+w][:] = 255
    		
        return output
            
    def __get_connected_components(self,img):
        img_bw = img.copy()
        img_bw = cv2.cvtColor(img_bw, cv2.COLOR_BGR2GRAY)
        _,img_bw = cv2.threshold(img_bw,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return self.__process_line(img_bw, img)
        
    def prepare(self,filename):
        #prepares the file (pdf preferably) for extraction i.e. forms vars
        if(self.__exists(filename)):
            self.current_file = filename
            self.images = convert_from_path(filename, 200)
            self.__preprocess()
    
    def clear(self):
        self.current_file = ''
        self.images = []
        self.signatures = []
                
    def get_OCR(self):
        #gets OCR with the help of pytessseract
        if(self.current_file == ''):
            raise ValueError('Current file cannot be empty for this operation.\
                             Kindly prepare the file first')
        else:
            s = ''
            for image in self.images:
                s = s + image_to_string(image) + '\n'
            return s
        
    def load_signature(self, image_path, info):
        if(self.__exists(image_path)):
            image = cv2.imread(image_path)
            self.signatures.append(signature(image,info))
            self.sig_hashes.append(imagehash.phash(Image.fromarray(image)))
        
    def extract(self):
        #get all the signatures in the document into one stuff
        #warning if signatures or the images are empty
        MIN_COMMON_AREA = 0.1
        NUM_TOP_MATCHES = 4
        self.payload = []
        sig_matches = [ [] for i in range(len(self.signatures)) ]
        sig_hash_matches = sig_matches.copy()
        sig_areas = sig_matches.copy()
        sig_diffs = sig_matches.copy()
        for img in self.images:
            #get the rects with high probability of having signatures
            rects = self.__get_selective_matches(img)
            for k, sig in enumerate(self.signatures):
                matches = []
                hash_matches = []
                areas = []
                hash_diffs = []
                det_rects = []
                #find the matching boxes
                for i, rect in enumerate(rects):
                    x, y, w, h = rect
                    if(self.__match(sig.sig,img[y:y+h,x:x+w])): 
                        matches.append(img[y:y+h,x:x+w])
                        det_rects.append(rect)
                #find the overlapping areas for each box with another
                #this box weeds out false positives
                if(det_rects == []):
                    areas = np.array(areas)
                elif(len(det_rects) > 1):
                    for i, rect in enumerate(det_rects[:-1]):
                        temp = []
                        for j in det_rects[i+1:]:
                            temp.append(self.__get_IOU(rect,j))
                        areas.append(np.sum(np.array(temp)))
                    #finding the positive boxes with greatest overlap    
                    matches = [matches[i] for i in np.argsort(areas)]
                    areas = np.sort(areas)
                    matches = [matches[i] for i in np.nonzero(areas > MIN_COMMON_AREA)[0]]
                    areas  = areas[np.nonzero(areas > MIN_COMMON_AREA)]
                    
                    hash_diffs = []
                    hash_matches = matches.copy()
                    
                    for i, rect in enumerate(hash_matches):
                        test_hash = imagehash.phash(Image.fromarray(rect))
                        hash_diffs.append(test_hash - self.sig_hashes[k])
                    hash_matches = [hash_matches[i] for i in np.argsort(hash_diffs)[::-1]]
                    
                else:
                    areas = np.array([MIN_COMMON_AREA])
                    x, y, w, h = det_rects[0]
                    test_hash = imagehash.phash(Image.fromarray(img[y:y+h,x:x+w]))
                    hash_diffs = [test_hash - self.sig_hashes[k]]
                    
                sig_matches[k] = sig_matches[k] + matches#[-NUM_TOP_MATCHES:]
                sig_areas[k] = sig_areas[k] + areas.tolist()#[-NUM_TOP_MATCHES:].tolist()
                sig_hash_matches[k] = sig_hash_matches[k] + hash_matches
                sig_diffs[k] = sig_diffs[k] + hash_diffs
        
        for k, sig in enumerate(self.signatures):
            #sig_matches[k] = [sig_matches[k][i] for i in np.argsort(sig_areas[k])]
            #sig_hash_matches[k] = [sig_hash_matches[k][i] for i in np.argsort(sig_diffs[k])[::-1]]
            sig_indices = np.argsort(sig_areas[k])
            sig_hash_indices = np.argsort(sig_diffs[k])
            p = 0.7
            agg_matches = []
            for i, indices in enumerate(sig_indices):
                agg_matches.append(p*np.where(sig_hash_indices==indices)[0][0] + (1-p)*i)
            new_matches = sig_matches[k].copy()
            new_matches = [new_matches[i] for i in np.argsort(agg_matches)]
#                    if((self.sig_hashes[k] - \
#                    imagehash.phash(Image.fromarray(sig_matches[k][i]))) < 30)]
            #this would be something like 
            #[[sig1, [match1, match2, ...]],[sig2, [match1, ...]], ...]
            self.payload.append([sig,new_matches])#[-NUM_TOP_MATCHES:]])
        return self.payload
    
class signature:
    def __init__(self, image, info):
        self.sig = image.copy()
        self.info = info