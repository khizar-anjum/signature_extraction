# -*- coding: utf-8 -*-

import os
import numpy as np
from pdf2image import convert_from_path 
from pytesseract import image_to_string
import imagehash
from PIL import Image
import cv2
import torch
from Model import SiameseConvNet, distance_metric
from Preprocessing import list_to_tensor

class extractor:
    """
    A class used to perform operations on scanned documents (pdf). It provides functions for signature detection as well as signature verification. It also provides an OCR utility.

    Attributes
    ----------
    current_file (str)
        name of the current pdf file under consideration
    batch_size (int)
        the batch size for forward pass in Siamese CNN. Adjust it according to your system's memory
    images (list)
        The pages of pdf file converted to images
    payload (list)
        A list of lists mentioning every signature and its closest matches
    sig_hashes (list)
        A list of stored hashes for signatures. This is done to avoid computing the hash anew everytime.

    Methods
    ------- 
    prepare(image_path)
        load the pdf file and convert it into several images
    load_signature(signature_image_path, info)
        load signature and its relevant info
    get_OCR()
        get OCR for the PDF file using pytesseract
    extract()
        extract the top matches for every signature loaded 
    clear()
        empty out all the inbuilt attributes
    """
    def __init__(self, batch_size = 8):
        # name of the current file under operation
        self.current_file = ''
        #batch size to be used. tinker it according to your needs
        self.batch_size = batch_size
        # list of the images a pdf file is converted to after preperation
        self.images = []
        self.signatures = []
        self.payload = []
        self.sig_hashes = []
        self.__load_model()
        
    def __load_model(self):
        device = torch.device('cpu')
        self.model = SiameseConvNet().eval()
        if(self.__exists('Models/model_epoch_2')):
            self.model.load_state_dict(torch.load('Models/model_epoch_2', map_location=device))
        
    def __exists(self,filename):
        if(os.path.exists(filename)):
            return True
        else:
            raise NameError('\''+filename+'\' does not exist in path.')
        
    def __match(self,sig,img_list):
        MAX_DIST = 0.20
        A = list_to_tensor([sig]*self.batch_size)
        match_list = []
        # a batch loop
        for i, x in enumerate(range(int(len(img_list)/self.batch_size)+1)):
            X = list_to_tensor(img_list[i*self.batch_size:(i+1)*self.batch_size])
            S = A[:X.shape[0]]
            
            f_A, f_X = self.model.forward(S, X)
            dist = distance_metric(f_A, f_X).detach().numpy()
            match_list.append(dist <= MAX_DIST)
        return match_list
            
    def __preprocess(self):
        for i,image in enumerate(self.images):
            image = np.array(image,dtype='uint8')
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
        GOOD_SIG = True
        _, img_org = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        img = np.stack([img]*3,axis=2)
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
        area_imgs = []
        for i, rect in enumerate(rects[:-1,...]):
            GOOD_SIG = True
            x, y, w, h = rect
            for j in rects[i+1:,...]:
                temp = self.__get_IOU(rect,j)
                if(temp > MAX_OVERLAP or h/w > 3): 
                    GOOD_SIG = False
                    break
            if GOOD_SIG:
                areas.append(rect)
                areas[-1] = (areas[-1]*frac).astype('int32')
                x, y, w, h = areas[-1]
                area_imgs.append(img_org[y:y+h,x:x+w])
        return areas, area_imgs
    
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
            if(w/h > 8 or (10000 < w*h and w*h < 20000)):
                output[y:y+h,x:x+w][:] = 255
    		
        return output
            
    def __get_connected_components(self,img):
        img_bw = img.copy()
        img_bw = cv2.cvtColor(img_bw, cv2.COLOR_BGR2GRAY)
        _,img_wb = cv2.threshold(img_bw,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return self.__process_line(img_wb, img_bw)
        
    def prepare(self,filename):
        """
        This function prepares the PDF for processing. i.e. it reads the pages from PDF and then stores them into images list. 
    
        Attributes
        ----------
        filename (str)
            name of the PDF file to be prepared
        """
        #prepares the file (pdf preferably) for extraction i.e. forms vars
        if(self.__exists(filename)):
            self.current_file = filename
            self.images = convert_from_path(filename, 200)
            self.__preprocess()
    
    def clear(self):
        """
        This function clears the class instance. 
        """
        self.current_file = ''
        self.images = []
        self.signatures = []
                
    def get_OCR(self):
        """
        This function gets the OCR of all the images loaded by reading the PDF file.
        
        Returns
        -------
        s (string)
        A string generated by OCR.         
        """
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
        """
        This function loads a signature (class) into signatures list. The signature image is preprocessed and stored for further usage. This will tell the program about what to look for in the image.
    
        Attributes
        ----------
        image_path (str)
            path to the signature image
        info (dict)
            a dictionary containing information about the signature. This can be anything like name or address of the person.
        """
        #convert a signature to grayscale and then threshold it
        if(self.__exists(image_path)):
            image = cv2.imread(image_path,0)
            _,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            self.signatures.append(signature(image,info))
            self.sig_hashes.append(imagehash.phash(Image.fromarray(image)))
        
    def extract(self):
        """
        This function extracts top matches of all the signatures loaded into a class instance, from the PDF file under consideration.
        
        Returns
        -------
        payload (list)
        contains all the top matches for signatures loaded in the instance. The format is somthing resembling the following list:
        [[sig1, [match1, match2, ...]],[sig2, [match1, ...]], ...]
        """        
        #get all the signatures in the document into one stuff
        #warning if signatures or the images are empty
        NUM_TOP_MATCHES = 4
        
        self.payload = []
        sig_matches = [ [] for i in range(len(self.signatures)) ]
        all_rects = []
        all_rect_imgs = []
        for img in self.images:
            #get the rects with high probability of having signatures
            rects, rect_imgs = self.__get_selective_matches(img)
            #skipping the first rect bbecause it is almost always the whole image
            all_rects = all_rects + rects[1:]
            all_rect_imgs = all_rect_imgs + rect_imgs[1:]

        #get some random signatures and match against them
        all_matches = np.zeros((len(all_rect_imgs),))
        ran_sig = np.random.choice(len(self.signatures), size=2, replace=False)
        for r in ran_sig:
            matches = self.__match(self.signatures[r].sig, all_rect_imgs)
            all_matches = np.logical_or(all_matches, np.hstack(matches))
        
        matched_sigs = [all_rect_imgs[i] for i,val in enumerate(all_matches) if val]
            
        for k, sig in enumerate(self.signatures):
            hash_diffs = []
            for sig_img in matched_sigs:
                test_hash = imagehash.phash(Image.fromarray(sig_img))
                hash_diffs.append(test_hash - self.sig_hashes[k])
            sig_matches[k] = [matched_sigs[i] for i in np.argsort(hash_diffs)][:NUM_TOP_MATCHES]
            self.payload.append([sig,sig_matches[k]])
        return self.payload
    
class signature:
    """
    A helper class to aid in saving of the signature image and relevant information such as signer's name. 

    Attributes
    ----------
    sig (numpy array)
        An image file
    info (dict)
        A dictionary for storing information
    """
    def __init__(self, image, info):
        self.sig = image.copy()
        self.info = info