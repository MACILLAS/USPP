import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input
from tensorflow.keras.losses import MeanAbsoluteError, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam

class MyNet(tf.keras.Model):
    def __init__(self, filters=100, hidden_layers=3):
        super(MyNet, self).__init__()
        self.filters = filters
        self.hidden_layers = hidden_layers

        self.conv3 = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')
        self.bn3 = BatchNormalization(epsilon=0.00001, momentum=0.1)

    def build(self, row, column, channels):
        inputs = Input(shape=(row, column, channels))
        x = inputs
        for _ in range(self.hidden_layers):
            x = Conv2D(filters=self.filters, kernel_size=3, strides=1, padding='same')(x)
            x = ReLU()(x)
            x = BatchNormalization(epsilon=0.00001, momentum=0.1)(x)

        x = self.conv3(x)
        x = self.bn3(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

@tf.function
def scrib_loss (stepsize_sim, loss1, stepsize_scr, loss2, stepsize_con, lhpy, lhpz):
    return stepsize_sim * loss1 + stepsize_scr * loss2 + tf.dtypes.cast((stepsize_con * (lhpy + lhpz)), tf.float32)

def saveMask (row, col, chn, im_target, name):
    # Define label_colours
    label_colours = np.random.randint(255, size=(100, 3))
    #im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
    im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(row, col, chn).astype(np.uint8)
    # Save mask (colourized)
    cv2.imwrite("output" + str(name) + ".png", im_target_rgb)

def segment(im, scribble=None, minLabels=3, nChannel=100, lr=0.01, stepsize_sim=1, stepsize_con=1, stepsize_scr=0.5, maxIter=500, hidden_layers=3):
    """
    This function takes an image along with the hyper-parameters and semantically segments the image.
    Returning a numpy array mask representing the different classes.
    Wrapper for mainProgram()
    :param im: (image) input image
    :param scribble: (image) the scribble mask. None if there is no mask
    :param minLabels: (int) the minimum amount of labels
    :param nChannel: (int) number of dims of CNN, good to start high
    :param lr: (float) learning rate
    :param stepsize_sim: (double)
    :param stepsize_con: (double)
    :param stepsize_scr: (double)
    :param maxIter: (int) max number of epochs to "train" the model
    :param hidden_layers (int) number of hidden layers
    :return: numpy segmentation mask
    """
    # Load Image
    data = np.array(im).astype('float32') / 255

    # load scribble
    if scribble is not None:
        #mask = cv2.imread(args.input.replace('.' + args.input.split('.')[-1], '_scribble.png'), -1)
        mask = scribble
        mask = mask.reshape(-1)
        mask_inds = np.unique(mask)
        mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
        inds_sim = np.where(mask == 255)[0]
        inds_scr = np.where(mask != 255)[0]
        target_scr = mask.astype(np.int)

        # set minLabels
        minLabels = len(mask_inds)

    # build model instance
    row, col, chn = data.shape
    #batch_size, row, col, chn = data.shape
    model = MyNet(filters=nChannel, hidden_layers=hidden_layers)
    model = model.build(row, col, chn)

    # set optimizer
    # opt = SGD(lr=args.lr, momentum=0.9)
    opt = Adam(lr=lr)
    # test Adam optimizer

    data = data.reshape((-1, row, col, chn))
    data = tf.convert_to_tensor(data, np.float32)
    dataset = tf.data.Dataset.from_tensor_slices([data])

    # We probably don't have to make these tensors but whatever
    #HPy_target = np.zeros(shape=(row-1, col, args.nChannel))
    HPy_target = tf.zeros([row - 1, col, nChannel], tf.float32) #added batch_size
    #HPz_target = np.zeros(shape=(row, col-1, args.nChannel))
    HPz_target = tf.zeros([row, col - 1, nChannel], tf.float32) #added batch_size

    # similarity loss definition
    loss_fn = SparseCategoricalCrossentropy()

    # scribble loss definition
    loss_fn_scr = SparseCategoricalCrossentropy()

    # continuity loss definition
    loss_hpy = MeanAbsoluteError()
    loss_hpz = MeanAbsoluteError()
    batch_idx = 0
    # loop over training steps
    for _ in dataset.repeat(maxIter):
        with tf.GradientTape() as tape:
            # forwarding
            # we should keep output as tensor otherwise gradients will not propagate
            output = model(data, training=True)[0]
            #output = model(data, training=True)
            outputHP = output
            output = tf.reshape(outputHP, [-1, nChannel])
            #output = tf.reshape(outputHP, [batch_size, -1, nChannel])

            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :] #changed to 2
            #HPy = outputHP[:, 1:, :, :] - outputHP[:, 0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            #HPz = outputHP[:, :, 1:, :] - outputHP[:, :, 0:-1, :]
            # lhpy = loss_hpy(HPy, HPy_target)
            lhpy = loss_hpy(HPy_target, HPy)
            # lhpz = loss_hpz(HPz, HPz_target)
            lhpz = loss_hpz(HPz_target, HPz)

            # convert this argmax to tensor
            # im_target = np.argmax(output, 1)
            im_target = tf.math.argmax(output, axis=1, output_type=tf.int32)  ##this is actually correct... (verfied from torch)
            im_target_debug = im_target.numpy()

            #nLabels = len(np.unique(im_target))  # this prob won't need to be converted (actually this work better)
            nLabels = len(np.unique(tf.cast(im_target, tf.float16)))

            if scribble is not None:
                # loss = args.stepsize_sim * loss_fn(tf.gather(im_target, inds_sim, axis=0), tf.gather(output, inds_sim, axis=0)) + args.stepsize_scr * loss_fn_scr(tf.gather(target_scr, inds_scr, axis=0), tf.gather(output, inds_scr, axis=0)) + tf.dtypes.cast((args.stepsize_con * (lhpy + lhpz)), tf.float32)
                loss = scrib_loss(stepsize_sim, loss_fn(tf.gather(im_target, inds_sim, axis=0), tf.gather(output, inds_sim, axis=0)),
                                  stepsize_scr, loss_fn_scr(tf.gather(target_scr, inds_scr, axis=0), tf.gather(output, inds_scr, axis=0)),
                                  stepsize_con, lhpy, lhpz)
            else:
                loss = stepsize_sim * loss_fn(im_target, output) + stepsize_con * (lhpy + lhpz)

            grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        #print(batch_idx, '/', args.maxIter, '|', ' label num :', nLabels)
        batch_idx += 1
        if nLabels <= minLabels:
            #print("nLabels", nLabels, "reached minLabels", minLabels, ".")
            break
    return im_target_debug.reshape(row, col).astype(np.uint8)

def mainProgram():
    # Load Image
    im = cv2.imread(args.input)
    data = np.array(im).astype('float32')/255

    # load scribble
    # scribble is 255 everywhere. Where there is a class its any number that is not 255.
    if args.scribble:
        mask = cv2.imread(args.input.replace('.' + args.input.split('.')[-1], '_scribble.png'), -1)
        mask = mask.reshape(-1)
        mask_inds = np.unique(mask)
        mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
        inds_sim = np.where(mask == 255)[0]
        inds_scr = np.where(mask != 255)[0]
        target_scr = mask.astype(np.int)

        # set minLabels
        args.minLabels = len(mask_inds)

    # build model instance
    row, col, chn = data.shape
    model = MyNet(filters=args.nChannel, hidden_layers=args.nConv)
    model = model.build(row, col, chn)
    tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)

    # set optimizer
    opt = Adam(lr=args.lr)

    data = data.reshape((-1, row, col, chn))
    data = tf.convert_to_tensor(data, np.float32)
    dataset = tf.data.Dataset.from_tensor_slices([data])

    # We probably don't have to make these tensors but whatever
    #HPy_target = np.zeros(shape=(row-1, col, args.nChannel))
    HPy_target = tf.zeros([row-1, col, args.nChannel], tf.float32)
    #HPz_target = np.zeros(shape=(row, col-1, args.nChannel))
    HPz_target = tf.zeros([row, col-1, args.nChannel], tf.float32)

    # similarity loss definition
    loss_fn = SparseCategoricalCrossentropy()

    # scribble loss definition
    loss_fn_scr = SparseCategoricalCrossentropy()

    # continuity loss definition
    loss_hpy = MeanAbsoluteError()
    loss_hpz = MeanAbsoluteError()
    batch_idx = 0
    # loop over training steps
    for _ in dataset.repeat(args.maxIter):
        with tf.GradientTape() as tape:
            # forwarding
            # we should keep output as tensor otherwise gradients will not propagate
            output = model(data, training=True)[0]

            outputHP = output
            output = tf.reshape(outputHP, [-1, args.nChannel])
            #output_reshaped_debug = output.numpy()

            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            #lhpy = loss_hpy(HPy, HPy_target)
            lhpy = loss_hpy(HPy_target, HPy)
            #lhpz = loss_hpz(HPz, HPz_target)
            lhpz = loss_hpz(HPz_target, HPz)

            # convert this argmax to tensor
            #im_target = np.argmax(output, 1)
            im_target = tf.math.argmax(output, axis=1, output_type=tf.int32) ##this is actually correct... (verfied from torch)
            #im_target = tf.math.reduce_max(output, axis=1, keepdims=False)
            im_target_debug = im_target.numpy()

            nLabels = len(np.unique(im_target)) # this prob won't need to be converted (actually this work better)
            nLabels = len(np.unique(tf.cast(im_target, tf.float16)))
            if args.visualize:
                # This does not work...
                print("Not Implemented Yet...")

            # loss
            # original code: loss_fn(output[inds_sim], target[inds_sim])
            # we need to switch the target and output for tensorflow
            # output: 170000x100 target_scr 170000,
            if args.scribble:
                # We can't access im_target like present because it is a tensor...
                # Best solution we have rn is tf.gather
                #loss = args.stepsize_sim * loss_fn(tf.gather(im_target, inds_sim, axis=0), tf.gather(output, inds_sim, axis=0)) + args.stepsize_scr * loss_fn_scr(tf.gather(target_scr, inds_scr, axis=0), tf.gather(output, inds_scr, axis=0)) + tf.dtypes.cast((args.stepsize_con * (lhpy + lhpz)), tf.float32)
                loss = scrib_loss(args.stepsize_sim, loss_fn(tf.gather(im_target, inds_sim, axis=0), tf.gather(output, inds_sim, axis=0)), args.stepsize_scr, loss_fn_scr(tf.gather(target_scr, inds_scr, axis=0), tf.gather(output, inds_scr, axis=0)), args.stepsize_con, lhpy, lhpz)
            else:
                loss = args.stepsize_sim * loss_fn(im_target, output) + args.stepsize_con * (lhpy + lhpz)

            grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        print(batch_idx, '/', args.maxIter, '|', ' label num :', nLabels)
        batch_idx += 1
        if nLabels <= args.minLabels:
            print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            break

    print('Outputting Image...')

    saveMask(row, col, chn, im_target, batch_idx)

def parse_args():
    # allow program to run here
    parser = argparse.ArgumentParser(description='Tensorflow Unsupervised Segmentation')
    parser.add_argument('--scribble', action='store_true', default=False,
                        help='use scribbles')
    parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                        help='number of channels')
    parser.add_argument('--maxIter', metavar='T', default=500, type=int,
                        help='number of maximum iterations') #1000
    parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                        help='minimum number of labels')
    parser.add_argument('--lr', metavar='LR', default=0.01, type=float,
                        help='learning rate')  # lr = 0.1
    parser.add_argument('--nConv', metavar='M', default=2, type=int,
                        help='number of convolutional layers')  # default 2
    parser.add_argument('--visualize', metavar='1 or 0', default=0, type=int,
                        help='visualization flag')
    parser.add_argument('--input', metavar='FILENAME', default='./test_data/aug/IMG_8159.JPG',
                        help='input image file name', required=False)
    parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                        help='step size for similarity loss', required=False)
    parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float,
                        help='step size for continuity loss')
    parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float,
                        help='step size for scribble loss')  # default 0.5
    # args = parser.parse_args()
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    mainProgram()
