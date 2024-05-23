import cv2
from ultralytics import YOLO

modelo = YOLO('modelos/pistola.pt')

video = cv2.VideoCapture('videos/pistol-6.mp4')

while True:
    check, img = video.read()
    img = cv2.resize(img,(1270,720))
    resultado = modelo.predict(img,verbose = False)

    for obj in resultado:
        nomes = obj.names
        # obj.show()
        for item in obj.boxes:
            x1,y1,x2,y2 = item.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cls = int(item.cls[0])
            nomeClasse = nomes[cls]
            conf = round(float(item.conf[0]),2)
            texto = f'{nomeClasse} - {conf}'
            cv2.putText(img,texto,(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),5)

    cv2.imshow('IMG',img)
    if cv2.waitKey(1) ==27:
        break



