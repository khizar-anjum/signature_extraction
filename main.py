# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 13:55:01 2019

@author: khizar
"""
from extractor import extractor
from tqdm import tqdm
from glob import glob
import cv2

ext = extractor()
ext.prepare('pdfs\\PoA_B14_Filipe Silva e Thore Kristiansen.pdf')

#%% loading some sigs
ext.load_signature('sigs\\A_2.png',{'name':'1st'})
ext.load_signature('sigs\\B_3.png',{'name':'second'})
ext.load_signature('sigs\\genuine-10.png',{'name':'third'})
ext.load_signature('sigs\\genuine-12.png',{'name':'fourth'})
#%%
ext.prepare('pdfs\\5_Scan_18022019_192748.pdf')
payload = ext.extract()
#%%
"""
for i,file in tqdm(enumerate(glob('pdfs\\*.pdf')[12:])):
    ext.prepare(file)
    payload = ext.extract()
    for possible in payload:
        for arrays in possible[1]:
            cv2.imwrite('imgs\\'+possible[0].info['name']+str(i+12)+'.jpg'\
                        ,arrays)
"""
 #%%