import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('best.pt')
    model.val(data='datasets/uw/uw.yaml',
              split='val',
              imgsz=640,
              batch=8,
              project='runs/val',
              name='exp'
              )