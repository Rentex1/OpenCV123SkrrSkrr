import numpy as np
import cv2 
import matplotlib as plt

Frau = cv2.imread('TestFrau.jpg')

#while True:


cv2.imshow('Test Imag', Frau) 
cv2.waitKey(0)
cv2.destroyAllWindows()
