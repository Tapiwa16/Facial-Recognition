#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:15:34 2020

@author: tapiwachikwenya
"""
import os

from PIL import Image

import numpy as np

import cv2

import pickle


face_cascade =cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#go to directory  of current py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#search for specific folder in similar path 
image_dir= os.path.join(BASE_DIR,"images")

#face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.EigenFaceRecognizer_create()

current_id= 0
label_ids= {}
x_train = []
y_labels= []

#iterate over files 
for root,dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpeg") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            #print(label, path)
            if not label in label_ids:
                label_ids[label]= current_id
                current_id +=1
            id_ = label_ids[label]
            print(label_ids) 
                
            #y_labels.append(label)#some number
            #x_train.append(path)# verify this image, turn into a NUMPY, array,GRAY
            pil_image= Image.open(path).convert("L")#grayscale
            size = (550,550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            #convert grayscale image into 
            image_array = np.array(pil_image,"uint8")
            #print(image_array) 
            faces =face_cascade.detectMultiScale(image_array, scaleFactor= 1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi= image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
#print(y_labels)
#print(x_train)
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)


recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")    

    
            
        