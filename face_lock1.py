import numpy as np
import cv2
import time

count = 0;

print("import clear")
faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')
print("recall model")
cap = cv2.VideoCapture(0)
cap.set(3,640) #set width
cap.set(4,480) #set Height
print("camera connected")

while True:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor = 1.5,
            minNeighbors=15,
            minSize=(25, 25),
        )
        if len(smile)==0:
           print("Fail(no smile)")
           count += 1 # Count Log-In Failure
           if count == 5:
               print("10초 후에 다시 시도하세요")
               for i in range(10):
                   time.sleep(1) #1 sec
                   print(10-i,'초...')
                   count = 0 # count init
                   continue
        else:
            for (xx, yy, ww, hh) in smile:
                cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 255), 2)
                print("Log-in")
                break
        
        cv2.imshow('video', img)
    k= cv2.waitKey(30) & 0xff
    if k == 27:
        break 
    
cap.release()
cv2.destroyAllWindows()