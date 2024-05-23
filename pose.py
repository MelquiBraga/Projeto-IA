import cv2
from ultralytics import YOLO

modelo = YOLO('yolov8n-pose.pt')

img = cv2.imread('img02.png')

resultado = modelo.predict(img,verbose = False)

for obj in resultado:
    # obj.show()
    pessoas = obj.keypoints.xy
    for pessoa in pessoas:
        for id,pontos in enumerate(pessoa):
            x,y = pontos
            x,y = int(x),int(y)
            print(x,y)
            cv2.circle(img,(x,y),4,(0,0,255),-1)
            cv2.putText(img,str(id),(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)

cv2.imshow('IMG',img)
cv2.waitKey(0)