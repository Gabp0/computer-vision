import cv2
import numpy as np

def extract_lines_from_chessboard(chessboard: np.ndarray) -> np.ndarray:

    grayscale_img = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(grayscale_img, (5, 5), 0)
    ret, otsu = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred_img, 20, 255)
    dilated_edges = cv2.dilate(edges, None, iterations=1)
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, threshold=100, minLineLength=75, maxLineGap=10)
    lines_img = np.zeros_like(chessboard)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lines_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return lines_img

def find_chessboard_squares(chessboard: np.ndarray) -> list:
    inverted_lines_img = cv2.cvtColor(cv2.bitwise_not(chessboard), cv2.COLOR_BGR2GRAY)
    board_contours, hierarchy = cv2.findContours(inverted_lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    square_centers = []
    for contour in board_contours:
        if 50 < cv2.contourArea(contour) < 5000:
            # Approximate the contour to a simpler shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Ensure the approximated contour has 4 points (quadrilateral)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = int((x + (x + w)) / 2)
                center_y = int((y + (y + h)) / 2)

                square_centers.append([center_x,center_y])

    return square_centers

def group_squares(square_centers: list) -> list:
    # Group into cols 
    y_sorted = sorted(square_centers, key=lambda y: y[1])
    board_rows = []
    current_row = [y_sorted[0]]
    for coord in y_sorted[1:]:
        if abs(coord[1] - current_row[-1][1]) < 10:  # Adjusted threshold for grouping
            current_row.append(coord)
        else:
            board_rows.append(current_row)
            current_row = [coord]
    board_rows.append(current_row)

    import random

    # Sort columns within each row
    for i in range(len(board_rows)):
        board_rows[i] = sorted(board_rows[i], key=lambda x: x[0])
        # drop a random column to test
        drop_idx = random.randint(0, len(board_rows[i]) - 1) if len(board_rows[i]) > 2 else None
        if drop_idx is not None:
            board_rows[i].pop(drop_idx)

    return board_rows

def fill_missing_squares(board_rows: list) -> list:

    # Fill missing squares
    for r in range(len(board_rows)):
        if len(board_rows[r]) < 8:
            row = board_rows[r]

            # Find the minimum x-difference between squares in the row
            min_x_diff = min(abs(row[j][0] - row[j + 1][0]) for j in range(len(row) - 1))
            # Find the average y-coordinate of the row
            avg_y = int(sum(coord[1] for coord in row) / len(row))

            # Check if its missing squares at the end of the row
            max_last_x = max(row[-1][0] for row in board_rows)
            last_square_diff = abs(row[-1][0] - max_last_x)
            while (last_square_diff > min_x_diff * 0.5) and len(board_rows[r]) < 8:
                new_square = [row[-1][0] + min_x_diff, avg_y]
                board_rows[r].append(new_square)
                last_square_diff = abs(board_rows[r][-1][0] - max_last_x)

            # Check if its missing squares at the start of the row
            min_first_x = min(row[0][0] for row in board_rows)
            first_square_diff = abs(row[0][0] - min_first_x)
            while (first_square_diff > min_x_diff * 0.5) and len(board_rows[r]) < 8:
                new_square = [row[0][0] - min_x_diff, avg_y]
                board_rows[r].insert(0, new_square)
                first_square_diff = abs(board_rows[r][0][0] - min_first_x)
                
            # Check if its missing squares in the middle of the row
            while len(board_rows[r]) < 8:
                row = board_rows[r]
                for i in range(1, len(row)): 
                    col_diff = row[i][0] - row[i - 1][0]
                    if abs(col_diff) > min_x_diff * 1.5:
                        new_square = [row[i - 1][0] + min_x_diff, avg_y]
                        board_rows[r].insert(i, new_square)
                        break

    return board_rows

def segment_chessboard(chessboard: np.ndarray) -> list:

    lines_img = extract_lines_from_chessboard(chessboard)
    square_centers = find_chessboard_squares(lines_img)
    board_rows = group_squares(square_centers)
    print(f"Found {len(board_rows)} rows with {len(board_rows[0])} squares each.")
    filled_board_rows = fill_missing_squares(board_rows)

    return filled_board_rows