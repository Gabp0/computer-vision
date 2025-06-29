from chesspiece_detector import ChessPieceDetector
from perspective_correction import correct_perspective
from checkerbord_segmentation import segment_chessboard

import cv2
import numpy as np
from sys import argv

MODEL_PATH =  "./chess-dectection-best-100epc.pt"

def assign_squares(chesspieces, board_rows):
    piece_squares = []

    for piece, (cx, cy) in chesspieces:
        closest_square = None
        for r, row in enumerate(board_rows):
            for c, square in enumerate(row):
                dist = np.sqrt((cx - square[0]) ** 2 + (cy - square[1]) ** 2)
                if closest_square is None or dist < closest_square[0]:
                    closest_square = (dist, r, c)

        if closest_square:
            dist, r, c = closest_square
            square_string = f"{chr(97 + r)}{1 + c}"
            piece_squares.append((piece, square_string))

    return piece_squares

if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: python chess_predictor.py <image_path>")
        exit(1)
    
    image = cv2.imread(argv[1])
    if image is None:
        print(f"Error: Could not read image from {argv[1]}")
        exit(1)
    
    piece_detector = ChessPieceDetector(MODEL_PATH)
    corrected_image = correct_perspective(image)
    chessboard_rows = segment_chessboard(corrected_image)
    chess_pieces = piece_detector.detect(corrected_image)
    piece_squares = assign_squares(chess_pieces, chessboard_rows)

    print("Detected chess pieces and their assigned squares:")
    for piece, square in piece_squares:
        print(f"{piece} at {square}")