from ultralytics import YOLO
import cv2
import numpy as np
from sys import argv
import matplotlib.pyplot as plt

IMG_SRC = argv[1]
MODEL_PATH = argv[2] if len(argv) > 2 else "./chess-dectection-best-100epc.pt"

class ChessDetector():
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
            
            # Get center of the chess piece bounding box
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            chess_pieces.append((chess_piece, (cx, cy))) 

        # Draw the chess pieces on the image
        for piece, (cx, cy) in chess_pieces:
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(image, piece, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Save the annotated image
        cv2.imwrite("output.png", image)

        return chess_pieces

if __name__ == "__main__":
    # Load the image
    image = cv2.imread(IMG_SRC)
    if image is None:
        print(f"Error: Could not read image from {IMG_SRC}")
        exit(1)


    detector = ChessDetector(MODEL_PATH)
    detected_pieces = detector.detect(image)

    print("Detected chess pieces:", detected_pieces)

    # Find chessboard area 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    