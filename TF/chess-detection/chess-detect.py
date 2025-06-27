from ultralytics import YOLO
import cv2
from sys import argv

IMG_SRC = argv[1]

model = YOLO("./chess-dectection-best-100epc.pt")  # load a pretrained model (recommended for training)

chessboard = cv2.imread(IMG_SRC)

results = model.predict(source=chessboard, iou=0.5, conf=0.5)

chess_pieces_names = model.names
for result in results[0].boxes:
    chess_piece = chess_pieces_names[int(result.cls.item())]  # class index of the detected object
    
    # Get the center of the bounding box
    x1, y1, x2, y2 = result.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    print(f"Detected {chess_piece} at position ({cx}, {cy})")

    cv2.circle(chessboard, (int(cx), int(cy)), 5, (0, 255, 0), -1)
    cv2.putText(chessboard, chess_piece, (int(cx), int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.imwrite("output.jpg", chessboard)