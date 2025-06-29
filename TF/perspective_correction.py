import cv2
import numpy as np

MIN_AREA_RATIO = 0.2        # minimum area ratio with respect to the image area for contour
POLY_EPISILON_RATIO = 0.1   # epsilon ratio for contour polygon approximation
CANNY_THRSH = (50, 150)     # Canny edge detection thresholds

def order_points(points: np.ndarray) -> np.ndarray:
    
    # calculate the sum and difference of the points
    points = points.reshape(4, 2)
    sum_points = points.sum(axis=1)
    diff_points = np.diff(points, axis=1)

    top_right = points[np.argmin(diff_points)]  # point with smallest difference
    top_left = points[np.argmin(sum_points)]    # point with smallest sum
    bottom_left = points[np.argmax(diff_points)] # point with largest difference
    bottom_right = points[np.argmax(sum_points)] # point with largest sum
    
    # return order of co-ordinates.
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

def calculate_target_points(points: np.ndarray) -> list[np.ndarray]:
    (top_left, top_right, bottom_right, bottom_left) = points

    # calculate the new width and height using euclidean distance
    left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))

    min_height = min(int(left_height), int(right_height))
    min_width = min(int(top_width), int(bottom_width))

    target_points = np.array([
        [0, 0],
        [min_width - 1, 0],
        [min_width - 1, min_height - 1],
        [0, min_height - 1]], dtype="float32")

    return target_points

def warp_square(frame: np.ndarray, ordered_points: np.ndarray, target_points: list[np.ndarray]) -> np.ndarray:
    transformation_matrix = cv2.getPerspectiveTransform(ordered_points, target_points)
    height = int(target_points[2][1] - target_points[0][1])
    width = int(target_points[1][0] - target_points[0][0])
    return cv2.warpPerspective(frame, transformation_matrix, (width, height))

def find_largest_square_contour(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_THRSH[0], CANNY_THRSH[1])
    dilated = cv2.dilate(edges, None, iterations=2)

    # Find largest square contour
    board_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    image_area = image.shape[0] * image.shape[1]
    for contour in board_contours:
        if cv2.contourArea(contour) < image_area * MIN_AREA_RATIO:
            continue

        poly = cv2.approxPolyDP(contour, POLY_EPISILON_RATIO * cv2.arcLength(contour, True), True)
        if len(poly) == 4:
            squares.append(poly)

    return sorted(squares, key=cv2.contourArea, reverse=True)[0]

def correct_perspective(image: np.ndarray) -> np.ndarray:

    largest_square = find_largest_square_contour(image)
    ordered_points = order_points(largest_square)
    target_points = calculate_target_points(ordered_points)

    return warp_square(image, ordered_points, target_points)
