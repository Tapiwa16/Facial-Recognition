#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:00:18 2020

@author: tapiwachikwenya"""


import cv2

camera = cv2.VideoCapture(0)
ret, img =camera.read()
cv2.imshow('img',img)
del(camera)
