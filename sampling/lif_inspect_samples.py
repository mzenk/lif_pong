# script for analyzing sampled data
from __future__ import division
from __future__ import print_function
import os
import sys
import yaml
import numpy as np
from scipy.ndimage import convolve1d
from lif_pong.utils import tile_raster_images
from lif_pong.utils.data_mgmt import make_figure_folder, get_rbm_dict
import lif_pong.training.rbm as rbm_pkg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = u'/home/hd/hd_hd/hd_kq433/ffmpeg-3.4.1-64bit-static/ffmpeg'


# video
def update_fig(i, frames, fig, img_artists, n_samples):
    if len(img_artists) > 1:
        assert len(frames) == len(img_artists)
    for idx, im in enumerate(img_artists):
        im.set_data(frames[idx][i])
    # artists[-1].set_text(
    fig.suptitle('Image {2}: {0:4d}/{1} samples'.format(
        i % n_samples, n_samples, (i // n_samples) % len(img_artists)), fontsize=14)
    return img_artists


def load_samples(sample_file, n_imgs, img_shape, n_labels, average=False,
                 show_probs=False, rbm=None, show_hidden=False, savename='test'):
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

    if show_probs and hid_samples.size > 0:
        if rbm is None:
            print('No RBM supplied for calculating probabilities.', file=sys.stderr)
        else:
            # marginal visible probabilities can be calculated from hidden states
            nh = hid_samples.shape[-1]
            vis_samples = rbm.sample_v_given_h(
                hid_samples.reshape(-1, nh))[0][:, :n_pixels]
            vis_samples = vis_samples.reshape(-1, n_samples, n_pixels)

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
        # # plot images for quick inspection
        # nrows = min(n_imgs, len(frames))
        # ncols = min(n_samples, 7)
        # snapshots = vis_samples[::n_imgs//nrows, ::n_samples//ncols].reshape(-1, *img_shape)
        # tiled_samples = tile_raster_images(snapshots,
        #                                    img_shape=img_shape,
        #                                    tile_shape=(nrows, ncols),
        #                                    tile_spacing=(1, 1),
        #                                    scale_rows_to_unit_interval=False,
        #                                    output_pixel_vals=False)

        # plt.figure(figsize=(14, 7))
        # plt.imshow(tiled_samples, interpolation='Nearest', cmap='gray')
        # plt.savefig(os.path.join(make_figure_folder(),
        #                          savename + '_snapshots.png'),
        #             bbox_inches='tight')
    return frames, n_samples


def make_video(frame_list, img_shape, n_samples, titles=[], show_hidden=False,
                savename='test'):
    if len(frame_list) == 1:
        fig, single_ax = plt.subplots()
        axarr = np.array([single_ax])
    else:
        tile_shape = (2, len(frame_list)//2 + len(frame_list) % 2)
        fig, axarr = plt.subplots(*tile_shape, figsize=(tile_shape[1]*6, tile_shape[0]*4.5))
        fig.subplots_adjust(wspace=0.1, hspace=0.3)
    if len(axarr.shape) == 1:
        axarr = np.expand_dims(axarr, 1)

    im = []
    for i, axrow in enumerate(axarr):
        for j, ax in enumerate(axrow):
            im.append(ax.imshow(np.zeros(img_shape), vmin=0, vmax=1.,
                      interpolation='Nearest', cmap='gray', animated=True))
            if len(titles) > 0:
                try:
                    title = titles[i*len(axrow) + j]
                    ax.set_title(title)
                except IndexError:
                    im.pop()
            else:
                ax.set_title('samples {}'.format(i*len(axrow) + j))
    ani = animation.FuncAnimation(
        fig, update_fig, frames=frame_list[0].shape[0],
        interval=10., blit=True, repeat=False,
        fargs=(frame_list, fig, im, n_samples))

    if show_hidden:
        ani.save(os.path.join(make_figure_folder(),
                              savename + '_hid.mp4'),
                 writer='ffmpeg')
    else:
        ani.save(os.path.join(make_figure_folder(),
                              savename + '.mp4'),
                 writer='ffmpeg')


def plot_samples(samples, tile_shape, img_shape, samples_per_tile=1000,
                 title='', savename='test'):
    n_pixels = np.prod(img_shape)
    n_samples = samples_per_tile*tile_shape[1]
    # needs (n_imgs, n_pxls); samples is (n_instances, n_samples, n_pixels)
    vis_samples = samples[..., :n_pixels]

    if tile_shape[0] != samples.shape[0]:
        if tile_shape[0] != samples.shape[1]:
            print('Wrong samples dimensions!', file=sys.stderr)
            sys.exit()
        else:
            vis_samples = np.swapaxes(vis_samples, 0, 1)
    if vis_samples.shape[1] > n_samples:
        print('Cropped some images (array too large)', file=sys.stderr)
    snapshots = vis_samples[:, :n_samples:samples_per_tile].reshape(-1, n_pixels)

    tiled_samples = tile_raster_images(snapshots,
                                       img_shape=img_shape,
                                       tile_shape=tile_shape,
                                       tile_spacing=(1, 0),
                                       # spacing_val=1.,
                                       scale_rows_to_unit_interval=False,
                                       output_pixel_vals=False)

    fig, ax = plt.subplots(figsize=(tile_shape[1]*2, tile_shape[0]*1.5))
    ax.imshow(tiled_samples, interpolation='Nearest', cmap='gray_r')
    ax.tick_params(left='off', right='off', bottom='off',
                   labelleft='off', labelright='off', labelbottom='off')
    fig.tight_layout()
    fig.savefig(os.path.join(make_figure_folder(), savename + '.png'))


def main(config_dict):
    sample_files = config_dict['sample_files']
    n_imgs = config_dict['n_imgs']
    img_shape = tuple(config_dict['img_shape'])
    n_labels = config_dict['n_labels']
    average = config_dict['average']
    show_probs = config_dict['show_probs']
    show_hidden = config_dict['show_hidden']
    make_video = config_dict['make_video']
    savename = config_dict['savename']
    if 'rbm_name' in config_dict.keys():
        rbm = rbm_pkg.load(get_rbm_dict(config['rbm_name']))
    else:
        rbm = None
    if 'titles' in config_dict.keys():
        titles = config_dict['titles']
    else:
        titles = []

    frame_list = []
    n_samples = []
    for i, f in enumerate(sample_files):
        tmp, ns = load_samples(
            f, n_imgs, img_shape, n_labels, average=average, 
            show_probs=show_probs, rbm=rbm,
            show_hidden=show_hidden, savename=savename + '_{:02d}'.format(i))
        frame_list.append(tmp)
        n_samples.append(ns)
    assert np.all(np.array(n_samples) == n_samples[0])
    n_samples = n_samples[0]

    if make_video:
        make_video(frame_list, img_shape, n_samples, titles=titles,
                   show_hidden=show_hidden, savename=savename)
    else:
        tile_shape = config_dict['tile_shape']
        samples_per_tile = config_dict['samples_per_tile']

        samples = np.array(frame_list, dtype=float).reshape(-1, n_samples, np.prod(img_shape))
        assert tile_shape[0] == len(samples)
        plot_samples(samples, tile_shape, img_shape, 
                     samples_per_tile=samples_per_tile, title='',
                     savename=savename)



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml that specifies'
              ' the configuration.')
        sys.exit()

    # load list of identifiers used for selecting data
    with open(sys.argv[1]) as f:
        config = yaml.load(f)

    main(config)
