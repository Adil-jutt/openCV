import cv2
import numpy as np


cap = cv2.VideoCapture('.\\assets\\face.mp4')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret,frame= cap.read()
    width=int(cap.get(3))
    height=int(cap.get(4))

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #(img, scaleFactor, minNeighbors, minSize, maxSize)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),3)
        roi_gray= gray[ y:y+w,x:x+w]
        roi_color= frame[y:y+h,x:x+w]
        eyes= eye_cascade.detectMultiScale(roi_gray,1.3,5)
        for(ex,ey,ew,eh) in eyes:
             cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(0,255,0),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
