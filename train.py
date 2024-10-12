import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/UWNet/UWNet.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='datasets/uw/uw.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD',
                project='runs/train',
                name='exp',
                )