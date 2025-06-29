from ultralytics import YOLO
import cv2
import numpy as np
from sys import argv
import matplotlib.pyplot as plt

class ChessPieceDetector():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.chess_pieces_names = self.model.names

    def detect(self, image, iou=0.1, conf=0.1):
        results = self.model.predict(source=image, 
                                    augment=True, 
                                    iou=iou, 
                                    conf=conf, 
                                    device='cuda:0')
        chess_pieces = []
        for result in results[0].boxes:
            chess_piece = self.chess_pieces_names[int(result.cls.item())]
            
            # Get bottom center of the bounding box
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cx = int((x1 + x2) // 2)
            cy = int(y2 - ((y2 - y1) * 0.25))  # Center at the bottom
            
            chess_pieces.append((chess_piece, (cx, cy))) 

        return chess_pieces
