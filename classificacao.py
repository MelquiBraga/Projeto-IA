import cv2
from ultralytics import YOLO

modelo = YOLO('yolov8n-cls.pt')

img = cv2.imread('img01.png')

resultado = modelo.predict(img,verbose = False)

for obj in resultado:
    nomes = obj.names
    # obj.show()
    # print(obj.probs)
    top5 = obj.probs.top5
    for item in top5:
        print(nomes[item])
    obj.show()