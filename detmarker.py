import numpy as np
import cv2
import cv2.aruco as aruco
import utils

def getArucoCorners (img=None):
    '''
    getArucoCorners takes rgb image, returns numpy array of corners of the aruco marker
    :param img: RBG (0-255)
    :return: ndarray shape:(4, 2) in pixel values
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParameters)
    #output = aruco.drawDetectedMarkers(image, corners)
    # save the detection
    #cv2.imwrite("output_det.png", output)

    # corners are clockwise from top-left corner (red box)
    x1 = (corners[0][0][0][0], corners[0][0][0][1])
    x2 = (corners[0][0][1][0], corners[0][0][1][1])
    x3 = (corners[0][0][2][0], corners[0][0][2][1])
    x4 = (corners[0][0][3][0], corners[0][0][3][1])
    return np.array([x1, x2, x3, x4])

def selectROI(img):
    r = cv2.selectROI("Select ROI (press enter when finished)", img, fromCenter=False)
    return r

def calc_transform(ref_img, img_to_transform):
    # pts_dst
    pts_ref = getArucoCorners(ref_img)
    # pts_src
    pts_to_transform = getArucoCorners(img_to_transform)
    h, status = cv2.findHomography(pts_to_transform, pts_ref)
    return h

def crop(img, roi):
    return img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

def transform(ref_img, img_to_transform, roi, h):
    '''
    Transform img by h then crop by roi
    :param ref_img: the reference image
    :param img_to_transform: the image to transform
    :param roi: selected region of interest from ref_img
    :param h: homography matrix to transform img to ref_img
    :return: trans_crop_img
    '''
    # Warp perspective
    temp = cv2.warpPerspective(
        img_to_transform, h, (ref_img.shape[1], ref_img.shape[0]))
    # Crop image
    return temp[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

def main (ref_img, img_to_transform):
    '''

    :param ref_img: reference image
    :param img_to_transform: image to transform
    :return: img_to_transform
    '''
    #pts_dst
    pts_ref = getArucoCorners(ref_img)
    #pts_src
    pts_to_transform = getArucoCorners(img_to_transform)

    h, status = cv2.findHomography(pts_to_transform, pts_ref)
    temp = cv2.warpPerspective(
        img_to_transform, h, (ref_img.shape[1], ref_img.shape[0]))

    cv2.imwrite("corrected_img_to_transform.png", temp)

if __name__ == "__main__":
    ref_img = cv2.imread("./test_data/IMG_8159.JPG", -1)
    img_to_transform = cv2.imread("./test_data/IMG_8160.JPG", -1)
    main(ref_img, img_to_transform)




