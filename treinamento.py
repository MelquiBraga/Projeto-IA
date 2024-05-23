from ultralytics import YOLO
def main():
    modelo = YOLO('yolov8n.pt')
    resultado = modelo.train(data='imagens/data.yaml', epochs=30, imgsz=640,workers=2)

if __name__== '__main__':
    main()