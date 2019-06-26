import numpy as np
import cv2 
import matplotlib as plt

Frau = cv2.imread('TestFrau.jpg')
grayFrau = cv2.cvtColor(Frau, cv2.COLOR_BGR2GRAY)
faceHaar = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
faces = faceHaar.detectMultiScale(grayFrau, scaleFactor=1.1, minNeighbors=5)


print('Frauen found: ', len(faces))
cv2.imshow('grayFrau', grayFrau) 




cv2.waitKey(0)
cv2.destroyAllWindows()