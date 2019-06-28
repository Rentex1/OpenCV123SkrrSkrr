import numpy as np
import cv2 as cv2
import matplotlib as plt

Frau = cv2.imread('3Frauen.jpg')
grayFrau = cv2.cvtColor(Frau, cv2.COLOR_BGR2GRAY)
faceHaar = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
faces = faceHaar.detectMultiScale(grayFrau, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(Frau, (x, y), (x+w, y+h), (0, 255, 0), 3)
print('Frauen found: ', len(faces))
cv2.imshow('Frau', Frau) 




cv2.waitKey(5000)
cv2.destroyAllWindows()