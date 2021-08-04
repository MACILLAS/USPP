import tensorflow as tf
import cv2
import numpy as np


if __name__ == "__main__":

    im_gt_gray = cv2.imread('2_gt_mask.png', cv2.IMREAD_GRAYSCALE)
    im_gt_gray = np.array(im_gt_gray)
    im_gt_gray [im_gt_gray > 0] = 255

    im_mask_gray = cv2.imread('2_outputavg_mask.png', cv2.IMREAD_GRAYSCALE)
    im_mask_gray = np.array(im_mask_gray)
    im_mask_gray [im_mask_gray < 150] = 0
    im_mask_gray [im_mask_gray > 200] = 255
    #cv2.imshow('Black and White Image', im_mask_gray)
    #cv2.waitKey()

    im_gt_gray = im_gt_gray.clip(min=0, max=1)
    im_mask_gray = im_mask_gray.clip(min=0, max=1)

    #cv2.imshow('name', im_gt_gray)

    m = tf.keras.metrics.MeanIoU(2, name=None, dtype=None)
    m.update_state(im_gt_gray, im_mask_gray[:,:])
    print(m.result().numpy())