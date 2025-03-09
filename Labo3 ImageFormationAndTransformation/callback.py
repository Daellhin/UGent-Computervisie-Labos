#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:31:47 2020

@author: David
"""

import cv2
import numpy as np

def onMouse(k, x, y, s, param):
    if k == cv2.EVENT_LBUTTONDOWN:
        param[0] += 1
        print(param[0])

aantal = [0]
# De vierkante haken zijn essentieel, hierdoor wordt de parameter die je
# meegeeft een mutable object. Zoniet, zal de originele waarde van aantal 
# niet veranderd zijn nadat de callback retourneert!

cv2.namedWindow('klik')
cv2.setMouseCallback('klik', onMouse, aantal)

leeg = np.zeros((100,100))

cv2.imshow('klik', leeg)
key = cv2.waitKey()
while(key != 27):
    key = cv2.waitKey()
cv2.destroyAllWindows()