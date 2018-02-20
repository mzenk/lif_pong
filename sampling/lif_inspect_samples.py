# script for analyzing sampled data
from __future__ import division
from __future__ import print_function
import os
import sys
import yaml
import numpy as np
from scipy.ndimage import convolve1d
from lif_pong.utils import tile_raster_images
from lif_pong.utils.data_mgmt import make_figure_folder, load_images, get_rbm_dict, get_data_path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = u'/home/hd/hd_hd/hd_kq433/ffmpeg-3.4.1-64bit-static/ffmpeg'


# video
def update_fig(i, frames, fig, img_artists, n_samples, n_imgs):
    if len(img_artists) > 1:
        assert len(frames) == len(img_artists)
    for idx, im in enumerate(img_artists):
        im.set_data(frames[idx][i])
    # artists[-1].set_text(
    fig.suptitle('Image {2}: {0:4d}/{1} samples'.format(
        i % n_samples, n_samples, (i // n_samples) % n_imgs), fontsize=14)
    return img_artists


def load_samples(sample_file, n_imgs, img_shape, n_labels, average=False,
                 show_hidden=False, savename='test'):
    n_pixels = np.prod(img_shape)
    with np.load(os.path.expanduser(sample_file)) as d:
        # samples.shape: ([n_instances], n_samples, n_units)
        samples = d['samples'].astype(float)[:n_imgs]
        if len(samples.shape) == 2:
            samples = np.expand_dims(samples, 0)
        n_samples = samples.shape[1]
        print('Loaded sample array with shape {}'.format(samples.shape))

    vis_samples = samples[..., :n_pixels]
    hid_samples = samples[..., n_pixels + n_labels:]

    # # marginal visible probabilities can be calculated from hidden states
    # nh = hid_samples.shape[-1]
    # vis_samples = rbm.sample_v_given_h(hid_samples.reshape(-1, nh))[0][:, :n_pixels]

    if average:
        # running average over samples
        kwidth = 10
        kernel = np.ones(kwidth)/kwidth
        vis_samples = convolve1d(vis_samples, kernel, axis=1)
        hid_samples = convolve1d(hid_samples, kernel, axis=1)

    if show_hidden:
        n_hidden = hid_samples.shape[2]
        img_shape = (20, int(np.ceil(n_hidden/20)))
        frames = hid_samples.reshape(-1, *img_shape)
        # plot mean activity for each image as a function of time
        mean_activity = hid_samples.mean(axis=2)
        plt.figure()
        for i, m in enumerate(mean_activity):
            plt.plot(np.linspace(0, 1, len(m)), m, alpha=.5,
                     label='Image {}/{}'.format(i, n_imgs))
        plt.legend()
        plt.xlabel('Experiment time [a.u.]')
        plt.ylabel('Mean activity of hidden units')
        plt.savefig(os.path.join(make_figure_folder(),
                                 savename + '_hid_act.png'))
    else:
        frames = vis_samples.reshape(-1, *img_shape)
        # plot images for quick inspection
        nrows = min(n_imgs, len(frames))
        ncols = min(n_samples, 7)
        snapshots = vis_samples[::n_imgs//nrows, ::n_samples//ncols].reshape(-1, *img_shape)
        tiled_samples = tile_raster_images(snapshots,
                                           img_shape=img_shape,
                                           tile_shape=(nrows, ncols),
                                           tile_spacing=(1, 1),
                                           scale_rows_to_unit_interval=False,
                                           output_pixel_vals=False)

        plt.figure(figsize=(14, 7))
        plt.imshow(tiled_samples, interpolation='Nearest', cmap='gray')
        plt.savefig(os.path.join(make_figure_folder(),
                                 savename + '_snapshots.png'),
                    bbox_inches='tight')
    return frames, n_samples


def main(config_dict):
    sample_files = config_dict['sample_files']
    n_imgs = config_dict['n_imgs']
    img_shape = tuple(config_dict['img_shape'])
    n_labels = config_dict['n_labels']
    average = config_dict['average']
    show_hidden = config_dict['show_hidden']
    savename = config_dict['savename']

    frame_list = []
    n_samples = []
    for i, f in enumerate(sample_files):
        tmp, ns = load_samples(
            f, n_imgs, img_shape, n_labels, average=average,
            show_hidden=show_hidden, savename=savename + '_{:02d}'.format(i))
        frame_list.append(tmp)
        n_samples.append(ns)
    assert np.all(np.array(n_samples) == n_samples[0])
    n_samples = n_samples[0]

    if len(frame_list) == 1:
        fig, single_ax = plt.subplots()
        axarr = np.array([single_ax])
    else:
        tile_shape = (2, len(frame_list)//2 + len(frame_list) % 2)
        fig, axarr = plt.subplots(*tile_shape, figsize=(tile_shape[1]*8, tile_shape[0]*6))
    if len(axarr.shape) == 1:
        axarr = np.expand_dims(axarr, 1)

    im = []
    for i, axrow in enumerate(axarr):
        for j, ax in enumerate(axrow):
            im.append(ax.imshow(np.zeros(img_shape), vmin=0, vmax=1.,
                      interpolation='Nearest', cmap='gray', animated=True))
            try:
                title = config_dict['titles'][i*len(axrow) + j]
                ax.set_title(title)
            except KeyError:
                ax.set_title('samples {}'.format(i*len(axrow) + j))
    ani = animation.FuncAnimation(
        fig, update_fig, frames=frame_list[0].shape[0],
        interval=10., blit=True, repeat=False,
        fargs=(frame_list, fig, im, n_samples, n_imgs))

    if show_hidden:
        ani.save(os.path.join(make_figure_folder(),
                              savename + '_hid.mp4'),
                 writer='ffmpeg')
    else:
        ani.save(os.path.join(make_figure_folder(),
                              savename + '.mp4'),
                 writer='ffmpeg')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml that specifies'
              ' the configuration.')
        sys.exit()

    # load list of identifiers used for selecting data
    with open(sys.argv[1]) as f:
        config = yaml.load(f)

    main(config)
