# -*- coding: utf-8 -*-

from extractor import extractor

ext = extractor()

# loading some sigs
ext.load_signature('sigs\\A_2.png',{'name':'1st'})
ext.load_signature('sigs\\B_3.png',{'name':'second'})
ext.load_signature('sigs\\genuine-10.png',{'name':'third'})
ext.load_signature('sigs\\genuine-12.png',{'name':'fourth'})

ext.prepare('pdfs\\5_Scan_18022019_192748.pdf')
#this gives me the matches for all the four above signatures, based on the document
payload = ext.extract()

