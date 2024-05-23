import cv2

video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
nome = input('Digite o nome do objeto')
id = 0

while True:
    check,img = video.read()

    cv2.imshow('IMG',img)
    if cv2.waitKey(1) & 0xff == ord('s'):
        cv2.imwrite(f'imagens/{nome} {id}.jpg',img)
        id+=1
        print(id)

    if id==30:
        break