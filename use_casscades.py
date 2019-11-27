import numpy as np
import cv2

#cascade = cv2.CascadeClassifier(PATH TO THE XML FILE OF CASCADE)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# img = cv2.imread(IMAGE PATH)
img = cv2.imread('a.jpg')

#CONVERT THE IMAGE TO GRAY SCALE - 1 CHANNEL REQUIRED FOR HAAR CASCADES
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#PASS THE SINGLE CHANNEL GRAY IMAGE TO DETECT MULTISCALE
detection = cascade.detectMultiScale(gray, 1.3, 5)

#THE LIST WILL CONTAIN MULTIPLE DETECTED OBJECTS - SO ITERATE THROUGH IT  
for (x,y,w,h) in detection:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
#WAIT KEY IS ALWAYS REQUIRED AFTER IMSHOW TO PAUSE THE FRAME
cv2.imshow('img',img)
cv2.waitKey(0)

#TO CLEAR THE MESS AFTER PROGRAM EXECUTIUON
cv2.destroyAllWindows()