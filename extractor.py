# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 21:51:00 2019

@author: khizar
"""

import os
from keras.models import load_model
from pdf2image import convert_from_path 
from pytesseract import image_to_string
import cv2
import numpy as np

class extractor:
    def __init__(self,neural_net):
        #used to load up the basic neural network and intialize class variables
        #self.load_neural_model(neural_net)
        # name of the current file under operation
        self.__current_file = ''
        # list of the images a pdf file is converted to after preperation
        self.images = []
        self.signatures = []
        self.payload = []
        
    def __exists(self,filename):
        if(os.path.exists(filename)):
            return True
        else:
            raise NameError('\''+filename+'\' does not exist in path.')
        
    def __match(self,imgs):
        MIN_MATCH_COUNT = 10
        #using the brisk descriptor to find out the matches
        all_matches = []
        brisk = cv2.BRISK_create()
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                table_number = 6, # 12
                key_size = 12,     # 20
                multi_probe_level = 1) #2
        search_params = dict(checks=50)   # or pass empty dictionary
        
        for org in self.signatures:    
            matched_sigs = []
            for img in imgs:
                # find the keypoints and descriptors with ORB
                _, des1 = brisk.detectAndCompute(img,None)
                _, des2 = brisk.detectAndCompute(org.sig,None)                
                #find matches using flann                
                flann = cv2.FlannBasedMatcher(index_params,search_params)
                matches = flann.knnMatch(des1,des2,k=2)
                
                # store all the good matches as per Lowe's ratio test.
                good = []
                matches = [x for x in matches if len(x) == 2]
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)
            
                if len(good)>MIN_MATCH_COUNT:
                    matched_sigs.append(img)
            all_matches.append([org,matched_sigs])
        return all_matches
            
    def __preprocess(self):
        for i,image in enumerate(self.images):
            img = np.array(image,dtype='uint8')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            img = cv2.adaptiveThreshold(img, 255, \
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
            gray = cv2.bitwise_not(img)
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
            self.images[i] = cv2.warpAffine(gray, M, (w, h)\
                ,flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
    def prepare(self,filename):
        #prepares the file (pdf preferably) for extraction i.e. forms vars
        if(self.__exists(filename)):
            self.__current_file = filename
            self.images = convert_from_path(filename, 200)
            self.__preprocess()
            
    def get_current_file(self):
        return self.__current_file
    
    def clear(self):
        self.__current_file = ''
        self.images = []
        self.model = None
        self.signatures = []
        self.payload = []
        
        
    def load_neural_model(self, neural_net):
        #loads a neural net
        if(self.__exists(neural_net)):
            try:
                self.model = load_model(neural_net)
            except Exception as e:
                print(e)
                
    def get_OCR(self):
        #gets OCR with the help of pytessseract
        if(self.__current_file == ''):
            raise ValueError('Current file cannot be empty for this operation.\
                             Kindly prepare the file first')
        else:
            s = ''
            for image in self.images:
                s = s + image_to_string(image) + '\n'
            return s
        
    def load_signature(self, image_path, info):
        if(self.__exists(image_path)):
            image = cv2.bitwise_not(cv2.threshold(cv2.imread(image_path,0)\
                ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
            self.signatures.append(signature(image,info))
        
    def extract(self):
        #get all the signatures in the document into one stuff
        for img in self.images:
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            prms = []
            for i,c in enumerate(contours):
                prms.append(cv2.arcLength(c,True))
            #getting the top 30 contours with high parameters
            indices = np.argsort(np.array(prms))[-30:]
            good_contours = [contours[i] for i in indices]
            good_imgs = []
            #this loop serves to extract potential sigs into good_imgs
            for c in good_contours:
                #making a convex hull around the signature. 
                black = np.zeros_like(img)
                hull = cv2.convexHull(c)
                cv2.drawContours(black, [hull], -1, (255, 255, 255), -1)
                masked = cv2.bitwise_and(img, img, mask = black)
                x, y, w, h = cv2.boundingRect(c)
                good_sig = masked[y:y+h,x:x+w]
                
                #removing horizontal straight lines 
                n = 17 #filter width
                linek = np.zeros((n,n),dtype=np.uint8)
                linek[int((n-1)/2),...]=1
                x=cv2.morphologyEx(good_sig, cv2.MORPH_OPEN, linek ,iterations=1)
                good_sig-=x
                good_imgs.append(good_sig)
                
        #matching signatures and the image
        #i want something that could be json encoded
        #this would be something like 
        #[[sig1, [match1, match2, ...]],[sig2, [match1, ...]], ...]
        self.payload = self.__match(good_imgs)
        return self.payload
    
class signature:
    def __init__(self, image, info):
        self.sig = image.copy()
        self.info = info