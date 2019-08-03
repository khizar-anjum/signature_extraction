# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 13:55:01 2019

@author: khizar
"""
from extractor import extractor

h = extractor('what is up')
h.prepare('pdfs/2_Scan_18022019_192700.pdf')

#%% loading some sigs
h.load_signature('sigs\\A_2.png',{'name':'jani no scene'})
h.load_signature('sigs\\B_3.png',{'name':'jani scene on'})

#%%
what = h.extract()



#next direction is to definitely use svm and autoencoder techniques to better 
#the signature matching