# -*- coding: utf-8 -*-
"""
Created on Sat May 19 20:09:43 2018

@author: kumar
"""

import cv2
import math as m
import numpy as np
import os
import pandas as pd

#Histogram of oriented gradient is used to find the gradient (intensity change at edges or corners)
#It gives the intensity change at leaf veins and corners
HOG = []

src = r"C:\Users\kumar\Desktop\Git_final_project\Folio_Leaf_Dataset\Folio"
source = r"C:\Users\kumar\Desktop\Git_final_project\Folio_Leaf_Dataset"
t = '\\'
count = 0
index = []

destSobelx = source+t+"Folio_Sobelx_Gray"
destSobely = source+t+"Folio_Sobely_Gray"

try:
    os.makedirs(destSobelx)
    os.makedirs(destSobely)
except:
    print("directory already exists")

dirs=os.listdir(src)
dirs.sort()

for i in dirs:
    dirs2=os.listdir(src+t+i)
    dirs2.sort()
    
    try:
        os.makedirs(destSobelx+t+i)
        os.makedirs(destSobely+t+i)
    except:
        print("folder already exists")
        
    for j in dirs2:
        index.append(count)
        name = src+t+i+t+j
        print(name)
        img = cv2.imread(name)
        img = cv2.GaussianBlur(img,(5,5),0)
        img = cv2.resize(img,(64,64))
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #Gradient across x axis
        sobelx = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=5)
        cv2.imwrite(destSobelx+t+i+t+j,sobelx)
        #Gradient change across y axis
        sobely = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=5)
        cv2.imwrite(destSobely+t+i+t+j,sobely)
        #finding magnitude and angle of gradient
        mag, angle = cv2.cartToPolar(sobelx,sobely,angleInDegrees=True)
        
        hog = []
        #scaling of 10 is done( angle change from 0to 360)
        #for more info read histogram of oriented gradient
        for pop in range(0,37):
            hog.append(0)
            
        hog = np.array(hog)
        
        for k in range(0,64):
            for l in range(0,64):
                x = angle[k][l]
                p = m.floor(x/10) * 10
                if p!=36:
                    q = p+10
                else:
                    q=0
                
                floor_a = x-p
                floor_b = 10-floor_a
                    
                hog[p//10] += floor_b * mag[k][l]
                hog[q//10] += floor_a * mag[k][l]
        HOG.append(hog)                           
    count+=1
df = pd.DataFrame(HOG)
df.index = index
df.to_csv('hog_gray.csv')