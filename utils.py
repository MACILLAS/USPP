import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore, mode
from segwscribb_dev import saveMask
from skimage import transform
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import (erosion, dilation, opening, closing, square)
import random
import imgaug.augmenters as iaa

def consensus_masks_ssim(list_masks, ssim_thresh=0.8):
    """
    This function is very similar to the consensus_mask that utilizes ssim instead of MSE.
    I think this function is definetly more reliable than MSE.
    In the future we should look into CATSIM...
    :param list_masks:
    :param ssim_thresh:
    :return:
    """
    mask_vects = list_masks

    mask_vect_len = len(mask_vects)
    # create a triangle matrix where the diagonal of each row is compared to remaining columns
    # ahh maybe not... This seems way too complicated for a short list.
    # IF mask_vect_len >= 100 then we will have to think about doing some optimization...
    inliers_array = np.zeros(mask_vect_len)

    for i in range(mask_vect_len):
        ssim_list = [ssim(mask_vects[i], vect, data_range=mask_vects[i].max() - mask_vects[i].min()) for vect in
                     mask_vects]
        ssim_array = np.array(ssim_list)
        inliers = sum(ssim_array > ssim_thresh) - 1
        inliers_array[i] = inliers

    # We now have a 1D array with inliers at each row.
    # Let's calculate the mode of this vector and assert that it should be greater than (1/3) of mask_vect_len
    # Otherwise return all False...

    inliers_mode = max_mode(inliers_array)
    inliers = np.sum(inliers_array == inliers_mode)
    if inliers_mode >= int(mask_vect_len / 3) and inliers >= int(mask_vect_len / 3):
        return inliers_array == inliers_mode
    else:
        return np.zeros(mask_vect_len, dtype=bool)


def consensus_masks(list_masks, mae_thresh=0.6):
    """
    This function takes a list of masks and returns the argwhere the masks agree
    :param list_masks: [[mask], [mask], [mask] ... ]
    :param mae_thresh: (float) this parameter is the threshold for inliers. This parameter will need to be adjusted depending on the dataset.
    :return: binary mask
    """
    # reshape each vector to 1D array (1 row)
    mask_vects = [np.reshape(mask, (1, -1)) for mask in list_masks]

    mask_vect_len = len(mask_vects)
    # create a triangle matrix where the diagonal of each row is compared to remaining columns
    # ahh maybe not... This seems way too complicated for a short list.
    # IF mask_vect_len >= 100 then we will have to think about doing some optimization...
    inliers_array = np.zeros(mask_vect_len)

    for i in range(mask_vect_len):
        mse_list = [mean_squared_error(mask_vects[i], vect) for vect in mask_vects]
        mse_array = np.array(mse_list)
        inliers = sum(mse_array < mae_thresh) - 1
        inliers_array[i] = inliers

    # We now have a 1D array with inliers at each row.
    # Let's calculate the mode of this vector and assert that it should be greater than (1/3) of mask_vect_len
    # Otherwise return all False...
    inliers_mode = max_mode(inliers_array)
    inliers = np.sum(inliers_array == inliers_mode)
    if np.average([inliers_mode, inliers]) > int(mask_vect_len / 3):
        return inliers_array == inliers_mode
    else:
        return np.zeros(mask_vect_len, dtype=bool)


def max_mode(inliers_array):
    '''
    simple program to return the maximum mode of a 1D-array
    :param inliers_array:
    :return: value of the maximum mode
    '''
    uniques = np.unique(inliers_array)
    assert uniques.shape[0] >= 1

    uniques_mode = np.zeros(uniques.shape[0])
    for i in range(uniques.shape[0]):
        count = np.sum(inliers_array == uniques[i])
        uniques_mode[i] = count

    comb_array = uniques_mode * uniques
    return uniques[np.argmax(comb_array)]


def mode_scribb(masks, percent=0.2):
    """
    Runs after consensus_mask or consensus_mask_ssim.
    This function randomly selects 20% of points in the mask.
    For a selected pixel, the mode of the pixel will become the scribble mask
    :param masks:
    :param percent:
    :return:
    """
    # get size of the mask (row, col, chn)
    row, col, chn = masks.shape
    # calculate the number of pixels we will use for scribbles
    num_rand_pix = int(row * col * percent)
    # get random numbers shape=(num_rand_pix, 2)
    rand_coord = np.random.rand(num_rand_pix, 2)
    # convert random numbers to range of coordinates
    rand_coord = np.floor(rand_coord * np.array([row, col])).astype(int)

    # get the values of masks at the rand_coord(s)
    values = masks[rand_coord[:, 0], rand_coord[:, 1], :]
    values_mode = mode(values, axis=1)
    # create bool array when the mode of the
    bool = values_mode[1] > (0.75 * chn)

    # create a boolean mask True where is part of percent chosen pixels and has same value between images
    scribb_bool = np.zeros((row, col), dtype=np.int8)
    scribb_bool[rand_coord[:, 0], rand_coord[:, 1]] = bool.reshape(-1)

    # create scribbles. We need to add one to differentiate 0 from background to 0 label class
    scribb = masks[:, :, 0]
    # set chosen pixels with the mode... (set them all, low consensus modes will be eliminated via scribb_bool)
    scribb[rand_coord[:, 0], rand_coord[:, 1]] = values_mode[0].reshape(-1)

    scribb = ((scribb + 1) * scribb_bool)
    scribb[scribb == 0] = 256
    scribb = scribb - 1
    return scribb


def consensus_scribb(masks, percent=0.3):
    """
    This is run after consensus_mask or consensus_mask_ssim.
    This function randomly selects 20% of points in the mask.
    If a point belong to the same class then it becomes part of the scribble mask

    The function returns the scribble mask
    :param masks: numpy array
    :param percent:
    :return:
    """
    # get size of the mask (row, col)
    row, col, _ = masks.shape
    # calculate the number of pixels we will use for scribbles
    num_rand_pix = int(row * col * percent)
    # get random numbers shape=(num_rand_pix, 2)
    rand_coord = np.random.rand(num_rand_pix, 2)
    # convert random numbers to range of coordinates
    rand_coord = np.floor(rand_coord * np.array([row, col])).astype(int)

    # get the values of masks at the rand_coord(s)
    values = masks[rand_coord[:, 0], rand_coord[:, 1], :]
    # create bool array when row is the same
    # added new condition (test without background class)
    bool_zero = values.max(axis=1) != 0
    bool_zero_choice = np.random.choice([0, 1], size=(num_rand_pix,), p=[0.8, 0.2])
    bool_zero = np.logical_xor(bool_zero, bool_zero_choice)

    bool = values.max(axis=1) == values.min(axis=1)
    bool = np.logical_and(bool, bool_zero)

    # create a boolean mask True where is part of percent chosen pixels and has same value between images
    scribb_bool = np.zeros((row, col), dtype=np.int8)
    scribb_bool[rand_coord[:, 0], rand_coord[:, 1]] = bool

    # create scribbles. We need to add one to differentiate 0 from background to 0 label class
    scribb = masks[:, :, 0]
    scribb = ((scribb + 1) * scribb_bool)
    scribb[scribb == 0] = 256
    scribb = scribb - 1
    return scribb


def process_raw_mask(mask):
    """
    This function takes the raw image mask as input from the network and corrects it.
    The numbering of the mask will be descending number of pixels in that class
    :param mask: mask numpy array
    :return: numpy array (In our formulation 0 (the largest) should be the background)
    """
    mask = mask
    # Find all the unique label values
    labels = np.unique(mask)
    # How many unique labels are there
    num_labels = len(labels)

    label_rank = pd.DataFrame(columns=['label_id', 'size'])
    for label in labels:
        label_mask = mask == label
        size = np.sum(label_mask)
        tempdf = pd.DataFrame(data=[[label, size]], columns=['label_id', 'size'])
        label_rank = label_rank.append(tempdf)

    label_rank = label_rank.sort_values('size', ascending=False)

    index = 0
    for row in label_rank.iterrows():
        label = row[1]['label_id']
        label_mask = mask == label
        mask[label_mask] = index
        index = index + 1

    return mask


def filter_img(img, filter="BLUR", **kwargs):
    """
    This utility takes a numerical mask and filters it using specified filter.
    The default filter is BLUR

    Future: Make these scipy.ndimage
    ie BLUR --> uniform_filter because cv2 is not returning decimals

    :param mask: binary mask ndarray
    :param filter: string {"BILATERAL", "BLUR", "GAUSSIAN", "MEDIANBLUR"}
    :param **kwargs: filter parameters as required
    :return: filtered mask
    """
    assert img is not None
    if filter == "BLUR":
        mask = cv2.blur(img, ksize=(kwargs['row'], kwargs['col']))  # this argument may not be correct
        mask = np.ceil(mask)
        return mask
    elif filter == "MEDIANBLUR":
        mask = cv2.medianBlur(img, kwargs['size'])
        return mask
    elif filter == "GAUSSIAN":
        mask = cv2.GaussianBlur(img, (5, 5), 0)
        return mask
    else:
        return img


def opening_mask(mask, size=3):
    """
    Morphological opening on an image is defined as an erosion followed by a dilation.

    :param mask:
    :param size:
    :return:
    """
    return opening(mask, square(size))

def rand_img_aug (img, prob=0.5, save=False, file_name="rand_aug.png"):
    """
    Takes an image and augmentation probability.
    A random image augmentation [BilateralBlur, EnhanceBrightness, AddToHueAndSaturation, Spatter, Autocontrast] is applied.
    :param img: Image to be transformed
    :param prob: The probability that the image will be transformed. The specific transformation is randomly choosen.
    :param save: Save the augmented image
    :param file_name: file_name to save the augmented image
    :return: augmented image
    """
    n_augs = 5 #currently supporting 5 different augmentations
    rand_n = random.random()
    if rand_n < prob:
        if rand_n < prob/n_augs:
            #print("Bilateral Blur")
            aug = iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))
            img = aug.augment_image(image=img)
        elif rand_n < 2 * (prob/n_augs):
            #print("Enhance Brightness")
            aug = iaa.pillike.EnhanceBrightness()
            img = aug.augment_image(image=img)
        elif rand_n < 3 * (prob/n_augs):
            #print("AddToHueAndSaturation")
            aug = iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
            img = aug.augment_image(image=img)
        elif rand_n < 4 * (prob/n_augs):
            #print("Spatter")
            aug = iaa.imgcorruptlike.Spatter(severity=(1, 2))
            img = aug.augment_image(image=img)
        else:
            #print("AutoContrast")
            aug = iaa.pillike.Autocontrast((10, 20), per_channel=True)
            img = aug.augment_image(image=img)
    if save:
        cv2.imwrite(file_name, img)
    return img

def single_img_aug_batch (img, batch_size=3):
    """
    For a single image create a batch where each image is the same image but subject to some augmentation.
    :param img: input image for copies
    :param batch_size: the number of copies
    :return: numpy array in shape [row, col, chn * batch_size]
    """
    batch_img = np.array(img)
    for _ in range(batch_size):
        aug_img = rand_img_aug(img, prob=1, save=False)
        aug_img = np.array(aug_img)
        batch_img = np.concatenate((batch_img, aug_img), axis=2)

    return batch_img