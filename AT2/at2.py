import cv2
from time import sleep
import numpy as np

def arrange_points(points):
    # initialize a list of co-ordinates that will be ordered
    # first entry is top-left point, second entry is top-right
    # third entry is bottom-right, forth/last point is the bottom left point.
    rectangle = np.zeros((4,2), dtype = "float32")
    
    # bottom left point should be the smallest sum
    # the top-right point will have the largest sum of point.
    sum_points= points.sum(axis =1)
    rectangle[0] = points[np.argmin(sum_points)]
    rectangle[2] = points[np.argmax(sum_points)]
    
    
    #bottom right will have the smallest difference
    #top left will have the largest difference.
    diff_points = np.diff(points, axis=1)
    rectangle[1] = points[np.argmin(diff_points)]
    rectangle[3] = points[np.argmax(diff_points)]
    
    
    # return order of co-ordinates.
    return rectangle

cap = cv2.VideoCapture('exemplo2.jpg')

ret, frame = cap.read()
h, w, _ = frame.shape

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = cv2.resize(frame, (w//5, h//5))
print(frame.shape)


#blurr the image
gray_scaled = cv2.GaussianBlur(frame,(5,5),0)

#Edge detection
edged = cv2.Canny(gray_scaled,50, 200)

# find contours in the edged image. keep only the largest contours.
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# loop over the contours.
squares = []
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    # approximate your contour
    approximation = cv2.approxPolyDP(contour, 0.02*perimeter, True)
    
    # if our contour has 4 points, then surely, it should be the paper.
    if len(approximation) != 4:
        continue

    squares.append(approximation)

# grab contours
# select contours based on size.
largest_square = sorted(squares, key=cv2.contourArea, reverse = True)[0]
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

rect_points = arrange_points(largest_square.reshape(4,2))
(top_left,top_right,bottom_right,bottom_left) = rect_points

left_height = np.sqrt(((top_left[0]-bottom_left[0])**2) + ((top_left[1]-bottom_left[1])**2))
right_height = np.sqrt(((top_right[0]-bottom_right[0])**2) + ((top_right[1]-bottom_right[1])**2))
top_width = np.sqrt(((top_right[0]-top_left[0])**2) + ((top_right[1]-top_left[1])**2))
bottom_width = np.sqrt(((bottom_right[0]-bottom_left[0])**2) + ((bottom_right[1]-bottom_left[1])**2))

maxheight = max(int(left_height), int(right_height))
maxwidth  = max(int(top_width), int(bottom_width))

destination = np.array([
    [0,0],
    [maxwidth -1,0],
    [maxwidth -1, maxheight-1],
    [0, maxheight - 1]], dtype = "float32")


matrix = cv2.getPerspectiveTransform(rect_points, destination)
warped = cv2.warpPerspective(rgb_frame, matrix, (maxwidth,maxheight))

# print(contours)

cv2.imshow('frame', warped)
cv2.waitKey(1)
sleep(100)