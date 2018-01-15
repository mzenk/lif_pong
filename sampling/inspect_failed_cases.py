from __future__ import division
from __future__ import print_function
import sys
import os
import yaml
import numpy as np
from lif_pong.utils import tile_raster_images
from lif_pong.utils.data_mgmt import make_figure_folder, load_images, make_data_folder
from cycler import cycler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

simfolder = '/work/ws/nemo/hd_kq433-data_workspace-0/experiment/simulations'

def show_imgs(expt_name):
    basefolder = os.path.join(simfolder, expt_name)
    folder_list = os.listdir(basefolder)
    with open(os.path.join(basefolder, folder_list[0], 'sim.yaml')) as f:
        simdict = yaml.load(f)
        data_name = simdict['general']['data_name']
        img_shape = simdict['general']['img_shape']
    wrong_idx = []
    for folder in folder_list:
        with open(os.path.join(basefolder, folder, 'wrong_cases')) as f:
            wrong_idx += list(yaml.load(f))
    print('Found {} wrong cases'.format(len(wrong_idx)))
    # load data and show up to 100 wrongly predicted cases in different plots
    _, _, test_set = load_images(data_name)
    for i in range(0, min(len(wrong_idx), 100), 25):
        imgs = tile_raster_images(test_set[0][wrong_idx[i:i+25]],
                                  img_shape=img_shape,
                                  tile_shape=(5, 5),
                                  tile_spacing=(1, 1),
                                  scale_rows_to_unit_interval=True,
                                  output_pixel_vals=False)

        plt.figure()
        plt.imshow(imgs, interpolation='Nearest', cmap='gray', origin='lower')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(make_figure_folder(),
                                 expt_name + '_wrong{}.png'.format(i//25)))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        expt_name = sys.argv[1]
    else:
        expt_name = 'GibbsPongWindowsSmall'
    show_imgs(expt_name)
