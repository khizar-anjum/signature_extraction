# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 21:51:00 2019

@author: khizar
"""

import os
from pdf2image import convert_from_path 
from pytesseract import image_to_string
from skimage import measure, morphology
from skimage.measure import regionprops
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
            self.images[i] = img
            
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
        MAX_OVERLAP = 0.70
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
        
        #filter upon the basis of maximum overlap allowed
        areas = []            
        for i, rect in enumerate(rects[:-1,...]):
            for j in rects[i+1:,...]:
                temp = self.__get_IOU(rect,j)
                if(temp > MAX_OVERLAP): 
                    areas.append(i)
                    break
        rects = np.delete(rects,obj=areas,axis=0)
        rects = rects*frac #resizing the reigons again
        rects = rects.astype('int32')
        return rects
            
    def __get_connected_components(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #block to increase contrast
        maxIntensity = 255.0 # depends on dtype of image data               
        # Parameters for manipulating image data
        phi = 1
        theta = 2
        newImage1 = (maxIntensity/phi)*(img/(maxIntensity/theta))**2
        newImage1 = np.array(newImage1,dtype='uint8')
        
        img = cv2.threshold(newImage1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # ensure binary
        # connected component analysis by scikit-learn framework
        blobs = img > img.mean()
        blobs_labels = measure.label(blobs, background=1)
        
        the_biggest_component = 0
        total_area = 0
        counter = 0
        average = 0.0
        for region in regionprops(blobs_labels):
            if (region.area > 10):
                total_area = total_area + region.area
                counter = counter + 1
            # take regions with large enough areas
            if (region.area >= 250):
                if (region.area > the_biggest_component):
                    the_biggest_component = region.area
        
        average = (total_area/counter)
        a4_constant = ((average/84.0)*150.0)
        
        b = blobs_labels.copy()
        component_sizes = np.bincount(b.ravel())
        too_small = np.logical_or((component_sizes < a4_constant),(component_sizes > 10000))
        too_small_mask = too_small[b]
        b[too_small_mask] = 0
        b = b.astype('uint8')
        b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #kernel = np.ones((3,3), np.uint8)
        #b = cv2.erode(b,kernel)
        return np.stack((b,b,b),axis=2)
        
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
        
    def extract(self):
        #get all the signatures in the document into one stuff
        #warning if signatures or the images are empty
        MIN_COMMON_AREA = 0.2
        NUM_TOP_MATCHES = 2
        self.payload = []
        sig_matches = [ [] for i in range(len(self.signatures)) ]
        sig_areas = sig_matches.copy()
        for img in self.images:
            #get the rects with high probability of having signatures
            rects = self.__get_selective_matches(self.__get_connected_components(img))
            for k, sig in enumerate(self.signatures):
                matches = []
                areas = []
                det_rects = []
                #find the matching boxes
                for rect in rects:
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
                else:
                    areas = np.array([MIN_COMMON_AREA])
                sig_matches[k] = sig_matches[k] + matches#[-NUM_TOP_MATCHES:]
                sig_areas[k] = sig_areas[k] + areas.tolist()#[-NUM_TOP_MATCHES:].tolist()
        
        for k, sig in enumerate(self.signatures):
            sig_matches[k] = [sig_matches[k][i] for i in np.argsort(sig_areas[k])]
            #this would be something like 
            #[[sig1, [match1, match2, ...]],[sig2, [match1, ...]], ...]
            self.payload.append([sig,sig_matches[k]])#[-NUM_TOP_MATCHES:]])
        return self.payload
    
class signature:
    def __init__(self, image, info):
        self.sig = image.copy()
        self.info = info