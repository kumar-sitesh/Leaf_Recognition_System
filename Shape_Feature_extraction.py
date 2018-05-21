# -*- coding: utf-8 -*-
"""
Created on Sat May 19 19:23:10 2018

@author: kumar
"""
#This code is for preprocessing and to extract Shape features like area,perimeter etc.
import cv2
import numpy as np
import pandas as pd
import os
#Path for leaf dataset 
src=r"C:\Users\kumar\Desktop\Git_final_project\Folio_Leaf_Dataset"
source= r"C:\Users\kumar\Desktop\Git_final_project\Folio_Leaf_Dataset\Folio"
t="\\"


dirs=os.listdir(source)
dirs.sort()

dest=src+t+"Folio_Box"
destOpen = src+t+"Folio_OPENING"
destCnt = src+t+"Folio_CONTOUR"
destHull = src+t+"Folio_HULL"

try:
    os.makedirs(dest)
    os.makedirs(destOpen)
    os.makedirs(destCnt)
    os.makedirs(destHull)
except:
    print("directory already exists")
    
species = []
no_of_iter=0
index = []
a=[]
p=[]
ar=[]
wr=[]
ptoa=[]
phulltopleaf=[]
ahtal=[]
count = 0

#Iterating through the root dataset folder
for i in dirs:
    dirs2=os.listdir(source+t+i)
    dirs2.sort()
    try:
        os.makedirs(dest+t+i)
        os.makedirs(destOpen+t+i)
        os.makedirs(destCnt+t+i)
        os.makedirs(destHull+t+i)
    except:
        print("folder already exists")
    sum_ar=0
    sum_wr=0
    sum_ptoa=0
    sum_phtp=0
    c_ar=0
    c_wr=0
    c_ptoa=0
    c_phtp=0 
    sum_ahtal=0
    c_ahtal=0
    
    #Iterating through each leaf species folder
    for j in dirs2:
        name=source+t+i+t+j
        
        #reading the image
        img=cv2.imread(name)
        
        #RESIZING image to size (512,512) pi
        img512 = cv2.resize(img,(512,512))
        img512_copy = cv2.resize(img,(512,512))
        img512_copy2 = cv2.resize(img,(512,512))
        #GRAYSCALING
        imggray=cv2.cvtColor(img512,cv2.COLOR_BGR2GRAY)
        
        #THRESHOLDING
        ret,imgthresh=cv2.threshold(imggray,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        #OPENING Operation
        kernel = np.ones((8,8),np.uint8)
        opening = cv2.morphologyEx(imgthresh, cv2.MORPH_OPEN, kernel)
        
        #INVERSE THRESHOLDING
        ret2,threshInv = cv2.threshold(opening,0,255,cv2.THRESH_BINARY_INV)
        
        #EROSION, to remove some of the pre-processing noise
        opening2 = cv2.erode(threshInv,kernel,iterations=1)
        cv2.imwrite(destOpen+t+i+t+j,opening2)
        
        #NORMALIZE image to Dtype=8U
        image = cv2.normalize(src=opening2, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
        
        #Finding CONTOURS
        ret,contours,hierarchy = cv2.findContours(image, 1, 2)
        cv2.drawContours(img512_copy,contours,-1,(0,0,254),3)
        cv2.imwrite(destCnt+t+i+t+j,img512_copy)
        
        #Finding AREA using Moments(using Contours)
        area=cv2.moments(contours[0])['m00']
        
        #Finding PERIMETER
        perimeter=cv2.arcLength(contours[0],True)
        
        #Estimating CONVEX HULL
        
        cnt = contours[0]
        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull)
        try:
            for k in range(defects.shape[0]):
                s,e,f,d = defects[k,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(img512_copy2,start,end,[0,0,254],2)
            cv2.imwrite(destHull+t+i+t+j,img512_copy2)
        except:
            print(i+j+"  : HULL not formed")
        
        hull1 = cv2.convexHull(contours[0])
        """cv2.drawContours(img512_copy2,hull1,-1,(0,0,254),3)
        cv2.imwrite(destHull+t+i+t+j,img512_copy2)"""
        #Finding Hull_Area and Hull_Perimeter
        hull_area = cv2.contourArea(hull1)
        hull_peri=cv2.arcLength(hull1,True)
        
        #Making a bounding rectangle using contours and estimating width and height of the box
        x,y,w,h = cv2.boundingRect(contours[0])
        cv2.rectangle(img512,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imwrite(dest+t+i+t+j,img512)
        
        #Checking if any values are noise generated, and adding a measure
        if hull_area==0:
            ahtal.append(0)
        else:
            ahtal.append(float(area)/hull_area)
            sum_ahtal+=float(area)/hull_area
            c_ahtal+=1
        if h==0:
            ar.append(0)
        else:
            ar.append((float(w)/h))
            sum_ar+=(float(w)/h)
            c_ar+=1
        if h==0 or w==0:
            wr.append(0)
        else:
            wr.append((float(area)/(w*h)))
            sum_wr+=(float(area)/(w*h))
            c_wr+=1
        if area<1000:
            ptoa.append(0)
        else:
            ptoa.append(float(perimeter)/area)
            sum_ptoa+=float(perimeter)/area
            c_ptoa+=1
        if perimeter<500:
            phulltopleaf.append(0)
        else:
            phulltopleaf.append(float(hull_peri)/perimeter)
            sum_phtp+=float(hull_peri)/perimeter
            c_phtp+=1
            
        index.append(count)
        no_of_iter+=1
    count+=1
    species.append(i)
    #Calculating MEAN of non-noise values of features and replacing the noise-generated values with the mean
    mean_ar=float(sum_ar)/c_ar
    mean_wr=float(sum_wr)/c_wr
    mean_ptoa=float(sum_ptoa)/c_ptoa
    mean_phtp=float(sum_phtp)/c_phtp
    mean_ahtal=float(sum_ahtal)/c_ahtal
    print(mean_ptoa,mean_phtp)
    for k in range(len(ar)):
        if ar[k]==0:
            ar[k]=mean_ar
    for k in range(len(wr)):
        if wr[k]==0:
            wr[k]=mean_wr
    for k in range(len(ptoa)):
        if ptoa[k]==0:
            ptoa[k]=mean_ptoa
    for k in range(len(phulltopleaf)):
        if phulltopleaf[k]==0:
            phulltopleaf[k]=mean_phtp
    for k in range(len(ahtal)):
        if ahtal[k]==0:
            ahtal[k]=mean_ahtal
            
result=[]
result.append(ar)
result.append(wr)
result.append(ptoa)
result.append(phulltopleaf)
result.append(ahtal)

#Creating and saving the dataframe to a CSV file
df=pd.DataFrame(result)
df.to_csv('result.csv')

spec = pd.DataFrame(species)
spec.to_csv('species.csv')
