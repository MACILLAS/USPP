import detmarker
import segwscribb_dev
import utils
import os
import cv2
import time
import numpy as np
from zaid_spall_lenz import load_data as defect_1b
from skimage.morphology import (erosion, dilation, opening, closing, square)

def prepDefect(REF_DIR = "./zaid_spall_lenz/defect_1b_subset/ref_frame", SUB_DIR = "./zaid_spall_lenz/defect_1b_subset/sub_frames", AUG_DIR = "./zaid_spall_lenz/defect_1b_subset/aug", scale=0.25):
    """
    This function is used to populate the aug directory in ./zaid_spall_lenz/defect_1b_subset

    :param REF_DIR: Directory contains the reference image
    :param SUB_DIR: Directory containing other images from different poses
    :param AUG_DIR: Directory where transformed images are stored.
    """

    ref_file = os.listdir(REF_DIR)[0]
    sub_file = os.listdir(SUB_DIR)

    ref_img = cv2.imread(os.path.join(REF_DIR, ref_file), -1)

    width = int(ref_img.shape[1] * scale)
    height = int(ref_img.shape[0] * scale)

    roi = detmarker.selectROI(ref_img)
    print(roi)
    #dim = (width, height)
    dim = (int(roi[2]*scale), int(roi[3]*scale))

    #cv2.imwrite(os.path.join(AUG_DIR, ref_file), cv2.GaussianBlur(cv2.resize(detmarker.crop(ref_img, roi), dim), (5, 5), 1))
    cv2.imwrite(os.path.join(AUG_DIR, ref_file), cv2.blur(cv2.resize(detmarker.crop(ref_img, roi), dim), (3, 3)))
    #cv2.imwrite(os.path.join(AUG_DIR, ref_file), cv2.resize(detmarker.crop(ref_img, roi), dim))

    for imgFile in sub_file:
        img = cv2.imread(os.path.join(SUB_DIR, imgFile), -1)
        img = detmarker.transform(ref_img, img, roi, defect_1b.calc_transform(ref_img, img))
        #cv2.imwrite(os.path.join(AUG_DIR, imgFile), cv2.GaussianBlur(cv2.resize(img, dim), (5, 5), 1))
        cv2.imwrite(os.path.join(AUG_DIR, imgFile), cv2.blur(cv2.resize(img, dim), (3, 3)))
        #cv2.imwrite(os.path.join(AUG_DIR, imgFile), cv2.resize(img, dim))

def prepTestData():
    '''
    This function is used to populate the aug directory in test_data
    '''
    REF_DIR = "./test_data/ref_frame"
    SUB_DIR = "./test_data/sub_frames"
    AUG_DIR = "./test_data/aug"
    ref_file = os.listdir(REF_DIR)[0]
    sub_file = os.listdir(SUB_DIR)

    ref_img = cv2.imread(os.path.join(REF_DIR, ref_file), -1)
    roi = detmarker.selectROI(ref_img)
    cv2.imwrite(os.path.join(AUG_DIR, ref_file), detmarker.crop(ref_img, roi))

    for imgFile in sub_file:
        img = cv2.imread(os.path.join(SUB_DIR, imgFile), -1)
        img = detmarker.transform(ref_img, img, roi, detmarker.calc_transform(ref_img, img))
        cv2.imwrite(os.path.join(AUG_DIR, imgFile), img)

def ensembleTestData():
    """
    ensembleTestData wraps up the ensemble method (makes the code easier)

    It saves all the images in the main directory
    :return:
    """
    #prepTestData()
    AUG_DIR = "./test_data/aug"
    # create list of files names in AUG_DIR
    aug_file = os.listdir(AUG_DIR)

    #ref_file = aug_file[0]
    #im = cv2.imread(os.path.join(AUG_DIR, ref_file), -1)

    #row, col, chn = im.shape
    #mask = segwscribb_dev.segment(im, scribble=None, minLabels=3, nChannel=100, lr=0.01, stepsize_sim=1, stepsize_con=1, stepsize_scr=0.5, maxIter=500)
    #mask = utils.process_raw_mask(mask)
    ##filtered_mask = utils.filter_mask(mask, 'MEDIANBLUR', size=1)

    # create list of images from list of file names in aug_file
    images = [cv2.imread(os.path.join(AUG_DIR, file), -1) for file in aug_file]
    # take the first image as the ref_file and use it to get the image dimensions (this should be the same for all augmented images)
    row, col, chn = images[0].shape

    start = time.time()
    # creates a list of masks from the list of images
    masks = [segwscribb_dev.segment(im, scribble=None, minLabels=3, nChannel=100, lr=0.01, stepsize_sim=1, stepsize_con=1, stepsize_scr=0.5, maxIter=200) for im in images]
    #masks = segwscribb_dev.segment(images, scribble=None, minLabels=3, nChannel=100, lr=0.001, stepsize_sim=1, stepsize_con=1, stepsize_scr=0.5, maxIter=500)
    masks = [utils.process_raw_mask(mask) for mask in masks]
    end = time.time()
    print(end - start)

    # Output Each Image for Debug Purpose
    segwscribb_dev.saveMask(304, 324, 3, masks[0], "_debug_0")
    segwscribb_dev.saveMask(304, 324, 3, masks[1], "_debug_1")
    segwscribb_dev.saveMask(304, 324, 3, masks[2], "_debug_2")
    segwscribb_dev.saveMask(304, 324, 3, masks[3], "_debug_3")
    segwscribb_dev.saveMask(304, 324, 3, masks[4], "_debug_4")
    segwscribb_dev.saveMask(304, 324, 3, masks[5], "_debug_5")
    segwscribb_dev.saveMask(304, 324, 3, masks[6], "_debug_6")

    # Returns index of 'good' masks
    # Assumes there are more good masks than bad masks.
    # And while good segs are similar, bad segs are different from one another.
    # It is possible that you will get all False which indicates you should run segmentation again.
    concensus = utils.consensus_masks(masks)

    # Check if concensus is all False
    if (~concensus).all():
        # indicate we need to rerun this function
        print("Segmentation Failure: Re-Run")

    concensus_mask = np.zeros((row, col, np.sum(concensus)))
    counter = 0
    for idx in np.argwhere(concensus):
        concensus_mask[:, :, counter] = masks[int(idx)]
        counter = counter + 1

    avg_mask = np.int_(np.round(np.average(concensus_mask, axis=2)))
    segwscribb_dev.saveMask(row, col, chn, avg_mask, "avg_mask")

def scribbsTestData():
    """
    Instead of using the inliers to create the ensemble mask.
    This program uses the inlier to create scribble (seed points)
    These seed points and images are iterated again through the system
    """
    # prepTestData()
    AUG_DIR = "./test_data/aug"
    # create list of files names in AUG_DIR
    aug_file = os.listdir(AUG_DIR)

    # create list of images from list of file names in aug_file
    images = [cv2.imread(os.path.join(AUG_DIR, file), -1) for file in aug_file]
    # take the first image as the ref_file and use it to get the image dimensions (this should be the same for all augmented images)
    row, col, chn = images[0].shape

    # scribbles mask is initially set to None
    scribbs = None
    # counts the number cycles
    cycle_counter = 0
    # base threshold for SSIM
    base_thresh = 0.8
    # thresh_step
    step = 0.03
    while True:
        # creates a list of masks from the list of images
        masks = [segwscribb_dev.segment(im, scribble=scribbs, minLabels=3, nChannel=100, lr=0.01, stepsize_sim=1,
                                        stepsize_con=1, stepsize_scr=0.5, maxIter=200) for im in images]
        # masks = segwscribb_dev.segment(images, scribble=None, minLabels=3, nChannel=100, lr=0.001, stepsize_sim=1, stepsize_con=1, stepsize_scr=0.5, maxIter=500)
        masks = [utils.process_raw_mask(mask) for mask in masks]

        # Output Each Image for Debug Purpose
        segwscribb_dev.saveMask(304, 324, 3, masks[0], "_debug_0")
        segwscribb_dev.saveMask(304, 324, 3, masks[1], "_debug_1")
        segwscribb_dev.saveMask(304, 324, 3, masks[2], "_debug_2")
        segwscribb_dev.saveMask(304, 324, 3, masks[3], "_debug_3")
        segwscribb_dev.saveMask(304, 324, 3, masks[4], "_debug_4")
        segwscribb_dev.saveMask(304, 324, 3, masks[5], "_debug_5")
        segwscribb_dev.saveMask(304, 324, 3, masks[6], "_debug_6")

        # Returns index of 'good' masks
        # Assumes there are more good masks than bad masks.
        # And while good segs are similar, bad segs are different from one another.
        # It is possible that you will get all False which indicates you should run segmentation again.
        concensus = utils.consensus_masks_ssim(list_masks=masks, ssim_thresh=(base_thresh + cycle_counter * step))
        # check if concensus is all False
        # if concensus is all false it reruns the loop
        if (~concensus).all():
            # indicate we need to rerun this function
            print("Segmentation Failure: Re-Run")
            continue
        else:
            cycle_counter = cycle_counter + 1

        if cycle_counter == 3:
            print("Max SSIM Thresh. Reached")
            break

        # Build the concensus_mask template
        concensus_mask = np.zeros((row, col, np.sum(concensus)))
        counter = 0
        for idx in np.argwhere(concensus):
            concensus_mask[:, :, counter] = masks[int(idx)]
            counter = counter + 1

        # concensus_mask is a numpy array of masks that are similar
        # scribbs = utils.consensus_scribb(concensus_mask, percent=0.2)
        scribbs = utils.consensus_scribb(concensus_mask, percent=(0.5 * cycle_counter))
        segwscribb_dev.saveMask(row, col, chn, scribbs.astype(int), "scribbles")

    # avg_mask = np.int_(np.round(np.average(concensus_mask, axis=2)))
    # segwscribb_dev.saveMask(row, col, chn, avg_mask, "avg_mask")
    print("Program Completed")

def scribbsDefect1B():
    """
    Instead of using the inliers to create the ensemble mask.
    This program uses the inlier to create scribble (seed points)
    These seed points and images are iterated again through the system
    """
    AUG_DIR = "./zaid_spall_lenz/defect_1b_subset/aug"
    # create list of files names in AUG_DIR
    aug_file = os.listdir(AUG_DIR)

    # create list of images from list of file names in aug_file
    images = [cv2.imread(os.path.join(AUG_DIR, file), -1) for file in aug_file]
    # take the first image as the ref_file and use it to get the image dimensions (this should be the same for all augmented images)
    row, col, chn = images[0].shape

    # scribbles mask is initially set to None
    scribbs = None
    # counts the number cycles
    cycle_counter = 0
    # base threshold for SSIM
    base_thresh = 0.6
    # thresh_step
    step = 0.05
    while True:
        # creates a list of masks from the list of images
        masks = [segwscribb_dev.segment(im, scribble=scribbs, minLabels=3, nChannel=100, lr=0.01, stepsize_sim=1,
                                        stepsize_con=1, stepsize_scr=0.5, maxIter=200) for im in images]
        # masks = segwscribb_dev.segment(images, scribble=None, minLabels=3, nChannel=100, lr=0.001, stepsize_sim=1, stepsize_con=1, stepsize_scr=0.5, maxIter=500)
        masks = [utils.process_raw_mask(mask) for mask in masks]

        # Output Each Image for Debug Purpose
        segwscribb_dev.saveMask(row, col, chn, masks[0], "_debug_0")
        segwscribb_dev.saveMask(row, col, chn, masks[1], "_debug_1")
        segwscribb_dev.saveMask(row, col, chn, masks[2], "_debug_2")
        segwscribb_dev.saveMask(row, col, chn, masks[3], "_debug_3")
        segwscribb_dev.saveMask(row, col, chn, masks[4], "_debug_4")
        segwscribb_dev.saveMask(row, col, chn, masks[5], "_debug_5")
        segwscribb_dev.saveMask(row, col, chn, masks[6], "_debug_6")
        segwscribb_dev.saveMask(row, col, chn, masks[7], "_debug_7")


        # Returns index of 'good' masks
        # Assumes there are more good masks than bad masks.

        # And while good segs are similar, bad segs are different from one another.
        # It is possible that you will get all False which indicates you should run segmentation again.
        concensus = utils.consensus_masks_ssim(list_masks=masks, ssim_thresh=(base_thresh + cycle_counter * step))
        # check if concensus is all False
        # if concensus is all false it reruns the loop
        if (~concensus).all():
            # indicate we need to rerun this function
            print("Segmentation Failure: Re-Run")
            continue
        else:
            cycle_counter = cycle_counter + 1

        if cycle_counter == 2:
            print("Max Cycle Reached.")
            break

        # Build the concensus_mask template
        concensus_mask = np.zeros((row, col, np.sum(concensus)))
        counter = 0
        for idx in np.argwhere(concensus):
            concensus_mask[:, :, counter] = masks[int(idx)]
            counter = counter + 1
        # concensus_mask is a numpy array of masks that are similar
        # scribbs = utils.consensus_scribb(concensus_mask, percent=0.2)
        scribbs = utils.consensus_scribb(concensus_mask, percent=(0.05 * cycle_counter))
        segwscribb_dev.saveMask(row, col, chn, scribbs.astype(int), "scribbles")

    avg_mask = np.int_(np.round(np.average(concensus_mask, axis=2)))
    avg_mask = utils.opening_mask(avg_mask, size=3)
    segwscribb_dev.saveMask(row, col, chn, avg_mask, "avg_mask")
    print("Program Completed")

    ### return the segmentation of reference frame ###
    #return avg_mask

def scribbsDefect(AUG_DIR = "./zaid_spall_lenz/defect_1b_subset/aug", scribbs=None):
    """
    Instead of using the inliers to create the ensemble mask.
    This program uses the inlier to create scribble (seed points)
    These seed points and images are iterated again through the system
    """
    # create list of files names in AUG_DIR
    aug_file = os.listdir(AUG_DIR)

    # create list of images from list of file names in aug_file
    images = [cv2.imread(os.path.join(AUG_DIR, file), -1) for file in aug_file]
    # take the first image as the ref_file and use it to get the image dimensions (this should be the same for all augmented images)
    row, col, chn = images[0].shape

    # scribbles mask is initially set to None
    #scribbs = None
    # counts the number cycles
    cycle_counter = 0
    # base threshold for SSIM
    base_thresh = 0.60
    # thresh_step
    step = 0.05
    while True:
        # creates a list of masks from the list of images
        masks = [segwscribb_dev.segment(im, scribble=scribbs, minLabels=3, nChannel=100, lr=0.001, stepsize_sim=1,
                                        stepsize_con=1, stepsize_scr=0.5, maxIter=200) for im in images]
        masks = [utils.process_raw_mask(mask) for mask in masks]

        # Output Each Image for Debug Purpose
        #segwscribb_dev.saveMask(row, col, chn, masks[0], "_debug_0")
        #segwscribb_dev.saveMask(row, col, chn, masks[1], "_debug_1")
        #segwscribb_dev.saveMask(row, col, chn, masks[2], "_debug_2")
        #segwscribb_dev.saveMask(row, col, chn, masks[3], "_debug_3")
        #segwscribb_dev.saveMask(row, col, chn, masks[4], "_debug_4")
        #segwscribb_dev.saveMask(row, col, chn, masks[5], "_debug_5")
        #segwscribb_dev.saveMask(row, col, chn, masks[6], "_debug_6")
        #segwscribb_dev.saveMask(row, col, chn, masks[7], "_debug_7")

        # Returns index of 'good' masks
        # Assumes there are more good masks than bad masks.

        # And while good segs are similar, bad segs are different from one another.
        # It is possible that you will get all False which indicates you should run segmentation again.
        concensus = utils.consensus_masks_ssim(list_masks=masks, ssim_thresh=(base_thresh + cycle_counter * step))
        # check if concensus is all False
        # if concensus is all false it reruns the loop
        if (~concensus).all():
            # indicate we need to rerun this function
            print("Segmentation Failure: Re-Run")
            continue
        else:
            cycle_counter = cycle_counter + 1

        if cycle_counter == 2:
            print("Max Cycle Reached.")
            break

        # Build the concensus_mask template
        concensus_mask = np.zeros((row, col, np.sum(concensus)))
        counter = 0
        for idx in np.argwhere(concensus):
            concensus_mask[:, :, counter] = masks[int(idx)]
            counter = counter + 1
        # concensus_mask is a numpy array of masks that are similar
        # scribbs = utils.consensus_scribb(concensus_mask, percent=0.2)
        scribbs = utils.consensus_scribb(concensus_mask, percent=0.1)
        #segwscribb_dev.saveMask(row, col, chn, scribbs.astype(int), "scribbles") ###DEBUG###

    avg_mask = np.int_(np.round(np.average(concensus_mask, axis=2)))
    avg_mask = np.clip(avg_mask, a_min=0, a_max=1)
    avg_mask = erosion(avg_mask)
    avg_mask = closing(closing(closing(avg_mask)))
    avg_mask = dilation(erosion(dilation(dilation(avg_mask))))

    segwscribb_dev.saveMask(row, col, chn, avg_mask, "avg_mask")
    print("Program Completed")

    ### return the segmentation of reference frame ###
    #return avg_mask

if __name__ == "__main__":

    scribbsDefect(AUG_DIR="./zaid_spall_lenz/defect_1b_subset/aug")

    #prepDefect(REF_DIR="./zaid_spall_lenz/defect_2_subset/ref_frame",
    #               SUB_DIR="./zaid_spall_lenz/defect_2_subset/sub_frames",
    #               AUG_DIR="./zaid_spall_lenz/defect_2_subset/aug", scale=0.35)

    #scribbsDefect(AUG_DIR = "./zaid_spall_lenz/defect_2_subset/aug", scribbs=None)


    #prepDefect(REF_DIR="./conestogo_spall_pensar/ref_frame",
    #               SUB_DIR="./conestogo_spall_pensar/sub_frames",
    #               AUG_DIR="./conestogo_spall_pensar/aug", scale=1)
    #scribbsDefect(AUG_DIR = "./conestogo_spall_pensar/aug", scribbs=None)
