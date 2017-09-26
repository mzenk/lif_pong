# Some useful functions
from __future__ import division
import numpy as np
import sys


# return a weighted average of the interval out_range for each vector in arr
# from skimage.measure import block_reduce could help here (and below)
def average_helper(out_range, arr):
    arr[np.all(arr == 0, axis=1)] = 1.
    return np.average(
        np.tile(range(out_range), (len(arr), 1)), weights=arr, axis=1)


# used by save_prediction_data scripts
def average_pool(arr, width, stride=None, offset=0):
    # array needs to have shape (n_imgs, n_t, n_pxls)
    if stride is None:
        stride = width
    kernel = np.ones(width) / width
    return np.apply_along_axis(
        np.convolve, 1, arr, kernel, 'valid')[:, int(offset)::int(stride)]
    # x = .5*(len(kernel) - 1)
    # filtered = convolve1d(
    #     arr, kernel, axis=1, mode='constant')[:, np.floor(x):-np.ceil(x)]
    # print((filtered == filtered1).mean())
    # doesn't give same result, but should be same and faster -> optimization


# this is more a reminder how it's done than an actual function
def make_chunks(data, n_chunks):
    n_each, remainder = data.shape[0] // n_chunks, data.shape[0] % n_chunks
    chunk_sizes = np.array([0] + [n_each] * n_chunks)
    chunk_sizes[1:(remainder + 1)] += 1
    chunk_ind = np.cumsum(chunk_sizes)
    return np.array_split(data, n_chunks), chunk_ind


# numerical stability
def logsum(x, axis=0):
    alpha = np.max(x, axis=axis) - np.log(sys.float_info.max)/2
    # alpha = np.max(x, axis=axis)
    return alpha + np.log(np.sum(np.exp(x - alpha), axis=axis))


def logdiff(x, axis=0):
    alpha = np.max(x, axis=axis) - np.log(sys.float_info.max)/2
    # alpha = np.max(x, axis=axis)
    return alpha + np.log(np.diff(np.exp(x - alpha), axis=axis)).squeeze()


def boltzmann(z, w, b):
    # unnormalised
    return np.exp(0.5 * np.dot(z.T, w.dot(z)) + np.dot(b, z))


def bin_to_dec(bin_array):
    length = bin_array.shape[1]
    base = np.power(2*np.ones(length), np.arange(0, length, 1)[::-1])
    return np.sum(bin_array*base, axis=1)


def dec_to_bin(dec_no, bits):
    if dec_no == 0:
        return np.zeros(int(bits))
    bin_array = np.zeros(int(bits))
    while dec_no > 0:
        exponent = np.floor(np.log2(dec_no))
        bin_array[int(bits - 1 - exponent)] = 1
        dec_no -= 2**exponent
    return bin_array


def compute_dkl(samples, p_target):
    dim = samples.shape[1]
    decimals = bin_to_dec(samples)
    p_sample = np.histogram(decimals, bins=np.arange(0, 2**dim + 1, 1),
                            normed=True)[0]
    p_sample_pos = p_sample.copy()
    p_sample_pos[p_sample_pos == 0] = 1
    dkl = np.sum(p_sample*np.log(p_sample_pos/p_target))
    return dkl


def to_1_of_c(labels, c):
    if np.min(labels) == 1:
        labels -= 1
    res = np.zeros((labels.size, c))
    res[range(labels.size), labels] = 1
    return res


def get_windowed_image_index(img_shape, end_index,
                             window_size=-1, fractional=False):
    assert end_index <= img_shape[1]
    mask = np.zeros(img_shape)
    if fractional:
        end_index = int(end_index * img_shape[1])
    else:
        assert isinstance(end_index, int)
    if window_size < 0:
        window_size = end_index
    start_index = max(0, end_index - window_size)
    mask[:, start_index:end_index] = 1
    uncovered_ind = np.nonzero(mask.flatten())[0]
    return uncovered_ind


# for MNIST --- make batches each containing equal number of labels
# not used yet. mnist has not exactly same number of train inst. for each class
def makebatches(data, targets, batchsize):
    assert data.shape[0] % batchsize == 0 and batchsize % 10 == 0
    sorting_ind = np.argsort(targets)
    sorted_data = data[sorting_ind]
    sorted_targets = targets[sorting_ind]
    n_batches = data.shape[0] / batchsize
    batchdata = sorted_data.reshape(batchsize, n_batches, data.shape[1])
    batchtargets = sorted_targets.reshape(batchsize, n_batches)
    return (np.swapaxes(batchdata, 0, 1), np.swapaxes(batchtargets, 0, 1))


# -----------------------------------------------------------------------------
# Taken from http://deeplearning.net/tutorial/utilities.html#how-to-plot
def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

# -----------------------------------------------------------------------------
