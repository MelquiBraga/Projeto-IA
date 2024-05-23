import cv2
import numpy as np
from ultralytics import YOLO

modelo = YOLO('yolov8l-seg.pt')

video = cv2.VideoCapture('people.mp4')

while True:
    check, img = video.read()

    resultado = modelo.predict(img,verbose = False)

    for obj in resultado:
        # obj.show()
        # print(obj.masks.data)
        for mask in obj.masks.data:
            maskConv = mask.cpu().numpy()
            # cv2.imshow('Mask',maskConv)
            maskConv = cv2.resize(maskConv,(img.shape[1],img.shape[0]))

            contours,_ = cv2.findContours(maskConv.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img,contours,-1,(0,255,0),3)

    cv2.imshow('IMG',img)
    if cv2.waitKey(1)==27:
        break