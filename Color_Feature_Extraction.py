# -*- coding: utf-8 -*-
"""
Created on Sat May 19 19:52:23 2018

@author: kumar
"""


import cv2
import pandas as pd
import os
#Color feature extraction using color histogram

L=[]
index=[]

src = r"C:\Users\kumar\Desktop\Git_final_project\Folio_Leaf_Dataset\Folio"
t = '\\'
count=0

dirs=os.listdir(src)
dirs.sort()
for i in dirs:
    dirs2=os.listdir(src+t+i)
    dirs2.sort()
    for j in dirs2:
        index.append(count)
        name = src+t+i+t+j
        print(name)
        image = cv2.imread(name)
        im1 = cv2.resize(image,(512,512))
        im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)
#Number Of pixels across each channel(blue,green,red) value ranging from 0 to 255
        hist1 = cv2.calcHist([im1],[0],None,[256],[0,255])
        hist2 = cv2.calcHist([im1],[1],None,[256],[0,255])
        hist3 = cv2.calcHist([im1],[2],None,[256],[0,255])

        l=[]
        for x in hist1:
            l.extend(x)
        for x in hist2:
            l.extend(x)
        for x in hist3:
            l.extend(x)
        #print(l)
        L.append(l)
    count+=1

df=pd.DataFrame(L,index=index)

column = []
for i in range(0,768):
    column.append('c'+str(i))

df.columns=column
df.to_csv('histo.csv')

df2 = pd.DataFrame(index)
df2.to_csv('index.csv')
