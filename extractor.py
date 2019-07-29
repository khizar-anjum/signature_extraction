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
        self.current_file = ''
        # list of the images a pdf file is converted to after preperation
        self.images = []
        
    def __exists(self,filename):
        if(os.path.exists(filename)):
            return True
        else:
            raise NameError('\''+filename+'\' does not exist in path.')
        
    def prepare(self,filename):
        #prepares the file (pdf preferably) for extraction i.e. forms vars
        if(self.__exists(filename)):
            self.current_file = filename
            self.images = convert_from_path(filename, 200)
            self.__preprocess()
            
    def __preprocess(self):
        for i,image in enumerate(self.images):
            img = np.array(image,dtype='uint8')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
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
            self.images[i] = cv2.warpAffine(gray, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
    def get_current_file(self):
        return self.current_file
    
    def clear(self):
        self.current_file = ''
        self.images = []
        self.model = None
        
    def load_neural_model(self, neural_net):
        #loads a neural net
        if(self.__exists(neural_net)):
            try:
                self.model = load_model(neural_net)
            except Exception as e:
                print(e)
                
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
        
    def extract(self):
        #get all the signatures in the document into one stuff
        pass