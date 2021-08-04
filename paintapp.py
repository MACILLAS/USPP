import math
import cv2
import numpy as np
import os

# This script is adapted from an OpenCV mouse tracking example...
# By: Satya Mallick @ https://learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
# Adapted by Max Midwinter, May 13, 2021
# The purpose of this script is to create "scribbles" for the unsupervised segmentation
# Currently you can only annotate up to three classes (RGB) (you won't need to annotate the background)
# Toggle 255 of RGB separately to annotate each class. Press esc when finished annotation.

# just change this...
imgdir = './conestogo_spall_pensar/aug'
imgname = 'frame860.jpg'


def nothing(x):
    pass


# Load an image
img = cv2.imread(os.path.join(imgdir, imgname))
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# Create trackbars for color change
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)

# Create trackbars for drawing shapes
cv2.createTrackbar('Select', 'image', 0, 3, nothing)

drawing = False  # true if mouse is pressed
ix, iy = -1, -1


# mouse callback function
def draw(event, x, y, flags, param):
    global ix, iy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if s == 3:
                cv2.circle(img, (x, y), 5, (b, g, r), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            if s == 0:
                cv2.rectangle(img, (ix, iy), (x, y), (b, g, r), 5)
            elif s == 1:
                cv2.circle(img, (int((ix + x) / 2), int((iy + y) / 2)),
                           int(math.sqrt(((ix - x) ** 2) + ((iy - y) ** 2))), (b, g, r), 5)
            elif s == 2:
                cv2.line(img, (ix, iy), (x, y), (b, g, r), 5)

        drawing = False


cv2.setMouseCallback('image', draw)

while 1:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    s = cv2.getTrackbarPos('Select', 'image')

cv2.destroyAllWindows()

scribb_mask = 255 * (np.ones_like(img)[:, :, 0])
for i in range(3):
    mask = img[:, :, i] == 255
    scribb_mask[mask] = i

cv2.imwrite(os.path.join(imgdir, "scribb_" + imgname +".png"), scribb_mask)
