import tensorflow as tf
import cv2
import numpy as np
from os import listdir
from os.path import isfile, isdir, join

def miou (im1, im2, classes=2):
    """
    :param im1: ground truth mask
    :param im2: predicted mask
    :param classes: the number of classes (i.e. foreground/background is 2 classes)
    :return: returns the result of the evaluation
    """
    m = tf.keras.metrics.MeanIoU(classes, name=None, dtype=None)
    m.update_state(im1, im2[:, :])
    return m.result().numpy()

def binary_preprocess (img1, img2):
    im_gt_gray = np.array(img1)
    #im_gt_gray[im_gt_gray < 150] = 1
    #im_gt_gray[im_gt_gray > 150] = 0
    im_gt_gray = np.clip(im_gt_gray, 0, 1)

    im_mask_gray = np.array(img2)
    im_mask_gray[im_mask_gray > 250] = 0
    im_mask_gray[im_mask_gray > 50] = 1
    im_mask_gray = np.clip(im_mask_gray, 0, 1)

    return im_gt_gray, im_mask_gray

def miou_from_file (gt_file, mask_file):
    im_gt_gray = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
    im_mask_gray = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    im_gt_gray, im_mask_gray = binary_preprocess(im_gt_gray, im_mask_gray)
    return im_gt_gray, im_mask_gray

def spall_sensitivity (folder_path="./spall_sensitivity"):
    """
    This function runs the sensitivity analysis for spalling images...

    :return:
    """
    gt_file_1b = './spall_sensitivity/1b_gt_mask.png'
    gt_file_2 = './spall_sensitivity/2_gt_mask.png'

    folder_path = folder_path
    onlydirs = [f for f in listdir(folder_path) if isdir(join(folder_path, f))]

    for dirs in onlydirs:
        print(dirs)
        onlyfiles = [i for i in listdir(join(folder_path, dirs)) if isfile(join(folder_path, dirs, i))]

        for file in onlyfiles:
            #print(file)
            if file[0] == "1":
                im_gt_gray, im_mask_gray = miou_from_file(gt_file_1b, join(folder_path, dirs, file))
                im_mask_gray = im_mask_gray[:228, :558]
                print(file + " " + str(miou(im_gt_gray, im_mask_gray[:, :], 2)))
            else:
                im_gt_gray, im_mask_gray = miou_from_file(gt_file_2, join(folder_path, dirs, file))
                im_mask_gray = im_mask_gray[:199, :384]
                print(file + " " + str(miou(im_gt_gray, im_mask_gray[:, :], 2)))

            #print(miou(im_gt_gray, im_mask_gray[1:, :], 2))

if __name__ == "__main__":

    #im_gt_gray, im_mask_gray = miou_from_file('./spall_sensitivity/1b_gt_mask.png', './spall_sensitivity/motblur_aug_severity_1/1b_motblur_10.png')
    #print (miou(im_gt_gray, im_mask_gray[1:, :], 2))
    spall_sensitivity()