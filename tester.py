# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:10:11 2019

@author: khizar
"""

from extractor import extractor



h = extractor('what is up')
h.prepare('pdfs/2_Scan_18022019_192700.pdf')
ocr = h.get_OCR()