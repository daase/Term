# 얼굴 인식과 웃음 인식 프로그램
import numpy as np
import cv2

print("카메라를 연결합니다")

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

print("카메라에 얼굴을 인식하세요")
cap = cv2.VideoCapture(0)
cap.set(3,640) #set width
cap.set(4,480) #set Height
print("얼굴이 인식되었습니다")

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(25, 25)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor = 1.5,
            minNeighbors=15,
            minSize=(30, 30),
        )
        if len(smile)==0: # 웃고 있지 않은 상태라면
           print("웃음을 인식할 수 없습니다") 
        else:  # 웃고 있는 상태라면
            print("웃음이 인식되었습니다") 
            for (xx, yy, ww, hh) in smile:
                cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 255), 2)
        
        cv2.imshow('video', img)
    k= cv2.waitKey(30) & 0xff
    if k == 27:
        break 
    
cap.release()
cv2.destroyAllWindows()