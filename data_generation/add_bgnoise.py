#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import sys
import os
import numpy as np
from lif_pong.utils.data_mgmt import load_images, make_data_folder


# method that accepts an image array [n_imgs, n_pixels] and adds Gaussian
# white noise on each. Clip at 0 and 1.
# => pxl(i) <- max(min(pxl(i) + e, 1), 0), where e~N(0, sigma)
def add_gaussian_noise(image_arr, sigma):
    noise = np.random.normal(scale=sigma, size=image_arr.shape)
    return np.clip(image_arr + noise, 0, 1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please specify the data_name (without datatype-ending).')
        sys.exit()

    # parameters that can be changed via command line
    data_name = sys.argv[1]
    sigma = .1
    train_data, valid_data, test_data = load_images(data_name)
    train_data[0] = add_gaussian_noise(train_data[0], sigma)
    valid_data[0] = add_gaussian_noise(valid_data[0], sigma)
    test_data[0] = add_gaussian_noise(test_data[0], sigma)

    np.savez_compressed(os.path.join(make_data_folder('datasets', True),
                                     data_name + '_bgnoise'),
                        (train_data, valid_data, test_data))
