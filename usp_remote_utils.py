import requests
import os
import cv2
from threading import Thread

def usp_request(url=None, image=None, scribble=None, minLabels=3, nChannel=100, lr=0.001, stepsize_sim=1, stepsize_con=1, stepsize_scr=0.5, maxIter=200):
    """
    This function sends POST request to url/predict to just segment the images

    :param url: url of web-service (local or remote)
    :param image: image in numpy.array
    :param scribble: scribble mask same dim as image but 1 channel also numpy.array
    :param minLabels: segmentation param, minimum num classes you want at convergence
    :param nChannel: number of classes to start at
    :param lr: the learning rate (ADAM)
    :param stepsize_sim: similarity loss factor
    :param stepsize_con: continuity loss factor
    :param stepsize_scr: scribble loss factor
    :param maxIter: maximum backprops (ie epochs)
    :return: segmentation as json
    """
    url = url
    if scribble is not None:
        scribble = scribble.tolist()

    data = {'image': image.tolist(), 'scribble': scribble, 'minLabels': minLabels, 'nChannel': nChannel, 'lr': lr,
            'stepsize_sim': stepsize_sim, 'stepsize_con': stepsize_con, 'stepsize_scr': stepsize_scr, 'maxIter': maxIter}
    r = requests.post(url, json=data)
    return r.json()


def main ():

    url = 'http://172.17.0.2:5000/predict'
    AUG_DIR = "./zaid_spall_lenz/defect_1b_subset/aug"

    # create list of files names in AUG_DIR
    aug_file = os.listdir(AUG_DIR)

    # create list of images from list of file names in aug_file
    images = [cv2.imread(os.path.join(AUG_DIR, file), -1) for file in aug_file]
    # take the first image as the ref_file and use it to get the image dimensions (this should be the same for all augmented images)
    row, col, chn = images[0].shape

    #masks = [segwscribb_dev.segment(im, scribble=scribbs, minLabels=3, nChannel=100, lr=0.001, stepsize_sim=1,
    #                                stepsize_con=1, stepsize_scr=0.5, maxIter=200) for im in images]
    im = images[0]

    ### Multi-Threaded Example
    '''
    threads = []
    for im in images:
       t = Thread(target=usp_request, args=(url, im, scribble))
       threads.append(t)
       t.start()
       
    for t in threads:
        t.join() 
    '''
    print(usp_request(url, im, None))

if __name__ == "__main__":
    main()