import cv2
import numpy as np
body_classifier=cv2.CascadeClassifier('haarcascade_fullbody.xml')

#video capture
cap=cv2.VideoCapture('walking.avi')

while cap.isOpened():
    ret,frame=cap.read()
    #captures each frame
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies=body_classifier.detectMultiScale(gray,1.2,3)
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(170,100,201),2)
        cv2.imshow('Pedestrians',frame)
    if cv2.waitKey(1)==13: #13 is the enter key
        break
cap.release()
cv2.destroyAllWindows()

