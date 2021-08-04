import os
import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation
from scipy.linalg import pinv
from matplotlib import pyplot as plt

# Use this script to load images/poses from HL2

def load_data():

    data_dir = 'defect_1b' # YOUR DATA FOLDER HERE

    poses_file = os.path.join(data_dir,'poses.csv')
    intrinsics_file = os.path.join(data_dir,'intrinsics.json')

    with open(intrinsics_file,'r') as f:
        intrinsics = json.load(f)

    poses = np.loadtxt(poses_file, delimiter=",")
    #P_mats = []
    RT_list =[]
    R_list = []
    K_list = []
    T_list = []
    for pose in poses:
        i = pose[0]
        transl = np.array(pose[1:4]).reshape(-1,1)

        q = pose[4:]
        r = Rotation.from_quat(q)
        Rot = r.as_matrix()
        K = np.array(intrinsics['camera_matrix'])

        im_file = os.path.join(data_dir,'images',str(int(i))+'.jpg')
        
        # Load image (I)
        I = cv2.imread(im_file) 
        if I is None:
            continue

        # Construct projection matrix (P)
        WorldTocameraMatrix = np.hstack([Rot, np.matmul(Rot, -1*transl)])
        P = K @ WorldTocameraMatrix
        K_list.append(K)
        R_list.append(Rot)
        T_list.append(transl)
        RT_list.append(WorldTocameraMatrix)
    return K_list, RT_list, R_list, T_list

def calc_transform(REF_img, OFF_img):
    """
    calc transform calculates a homography between the OFF_img and the REF_img
    the homography is used to transform the OFF_img to the REF_img

    This function uses cv2 functions SHIFT Feature Extractor, FLANN matcher, RANSAC and findHomography
    :param REF_img:
    :param OFF_img:
    :return: homography matrix H
    """
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(OFF_img, None)
    kp2, des2 = sift.detectAndCompute(REF_img, None)

    MIN_MATCH_COUNT = 10

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

    return M

if __name__ == '__main__':
    K_list, RT_list, R_list, T_list = load_data()
    data_dir = 'defect_1b' # YOUR DATA FOLDER HERE

    REF_img = cv2.imread(os.path.join(data_dir, 'images', '108.jpg'), -1)
    OFF_img = cv2.imread(os.path.join(data_dir, 'images', '24.jpg'), -1)

    im_dest = cv2.warpPerspective(OFF_img, calc_transform(REF_img, OFF_img), (OFF_img.shape[1], OFF_img.shape[0]))
    cv2.imwrite("corrected_img_to_transform.png", im_dest)








