import cv2
import numpy as np

CAP_SOURCE = 0  # camera source

GAUSSIAN_KERNEL = (5, 5)    # kernel size for Gaussian blur and dilation
CANNY_THRSH = (30, 200)     # Canny edge detection thresholds
MIN_AREA_RATIO = 0.2        # minimum area ratio with respect to the image area for contour
POLY_EPISILON_RATIO = 0.1   # epsilon ratio for contour polygon approximation

def contour_to_point_dist(countour: np.ndarray, point: tuple) -> int:

    # find contour center
    M = cv2.moments(countour)
    if M["m00"] == 0:
        return np.inf

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return int(np.sqrt((cx - point[0]) ** 2 + (cy - point[1]) ** 2))

def extract_borders(frame: np.ndarray) -> np.ndarray:
    # preprocess
    value_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    blurred = cv2.GaussianBlur(value_frame, GAUSSIAN_KERNEL, 0)
    
    # extract borders
    borders = cv2.Canny(blurred, CANNY_THRSH[0], CANNY_THRSH[1])
    dialation_shape = cv2.getStructuringElement(cv2.MORPH_CROSS, GAUSSIAN_KERNEL)
    dilated = cv2.dilate(borders, dialation_shape, iterations=1)

    return dilated

def find_square_contours(frame: np.ndarray) -> list[np.ndarray]:
    h, w = frame.shape
    img_area = h * w

    contours, _ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for contour in contours:
        # remove small contours
        if cv2.contourArea(contour) < img_area * MIN_AREA_RATIO:
            continue
        
        # get square like contours
        approx = cv2.approxPolyDP(contour, POLY_EPISILON_RATIO * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            squares.append(approx)

    return squares

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

def display_frame(frame: np.ndarray, square: np.ndarray, warped_square: np.ndarray) -> None:
    rf = 1 # resize factor
    h, w, _ = frame.shape
    display_frame = cv2.drawContours(frame, [square], -1, (0, 255, 0), 3)
    display_frame = cv2.resize(display_frame, (int(w / rf), int(h / rf)))
    cv2.putText(display_frame, "Press 'q' to save and quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if warped_square is not None:
        # resize warped square
        ws_h, ws_w, _ = warped_square.shape
        new_h = int(h / rf)
        new_w = int(ws_w * (new_h / ws_h))
        warped_square = cv2.resize(warped_square, (new_w, new_h))

        display_frame = cv2.hconcat([display_frame, warped_square])
    
    cv2.imshow('frame', display_frame)

def capture_loop() -> None:
    # Initialize camera
    cap = cv2.VideoCapture(CAP_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source '{CAP_SOURCE}'")

    # Capture loop
    central_square_points, warped_square = None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Unable to read from camera source '{CAP_SOURCE}'")
        
        h, w, _ = frame.shape

        border_frame = extract_borders(frame)
        squares = find_square_contours(border_frame)

        # sort squares by proximity to center
        squares = sorted(squares, key=lambda square: contour_to_point_dist(square, (w // 2, h // 2)))
        if len(squares) > 0:
            central_square_points = squares[0]

        if central_square_points is not None:
            # find points for transformation
            ordered_points = order_points(central_square_points)
            target_points = calculate_target_points(ordered_points)
            
            warped_square = warp_square(frame, ordered_points, target_points)

        # draw and show frame
        display_frame(frame, central_square_points, warped_square)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            
            # save the warped document
            if warped_square is not None:
                cv2.imwrite('document.jpg', warped_square)
                cv2.imshow('Warped Document', warped_square)
                cv2.waitKey(0)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_loop()
