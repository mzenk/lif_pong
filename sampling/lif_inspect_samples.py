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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
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


def load_samples(sample_file, data_idx, img_shape, n_labels, average=False,
                 show_probs=False, rbm=None, show_hidden=False, savename='test'):
    n_pixels = np.prod(img_shape)
    try:
        n_imgs = len(data_idx)
    except TypeError:
        n_imgs = data_idx
        data_idx = np.arange(data_idx)

    with np.load(os.path.expanduser(sample_file)) as d:
        # samples.shape: ([n_instances], n_samples, n_units)
        samples = d['samples'].astype(float)[data_idx]
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


def plot_animation(frame_list, img_shape, n_samples, titles=[], show_hidden=False,
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
                 offset=0, titles=[], savename='test'):
    n_pixels = np.prod(img_shape)
    n_samples = samples_per_tile*tile_shape[1] + offset
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
    snapshots = vis_samples[:, offset:n_samples:samples_per_tile].reshape(-1, n_pixels)

    tiled_samples = tile_raster_images(snapshots,
                                       img_shape=img_shape,
                                       tile_shape=tile_shape,
                                       tile_spacing=(1, 1),
                                       # spacing_val=1.,
                                       scale_rows_to_unit_interval=False,
                                       output_pixel_vals=False)

    fig, ax = plt.subplots(figsize=(tile_shape[1]*2, tile_shape[0]*1.5))
    ax.imshow(tiled_samples, interpolation='Nearest', cmap='gray_r')
    ax.tick_params(left='off', right='off', bottom='off', top='off',
                   labelleft='off', labelbottom='off')
    if len(titles) > 0:
        ax.tick_params(labelleft='on')
        imgheight = (tiled_samples.shape[0] + 1)/tile_shape[0]
        tick_locs = .5*imgheight + np.arange(tile_shape[0])*imgheight
        plt.yticks(tick_locs, titles, rotation='horizontal')

    fig.tight_layout()
    fig.savefig(os.path.join(make_figure_folder(), savename + '.png'))


def plot_samples_clamp(samples, tile_shape, img_shape, samples_per_tile=1000,
                       offset=0, titles=[], savename='test', clamp_duration=None,
                       clamp_window=None):
    n_pixels = np.prod(img_shape)
    n_samples = samples_per_tile*tile_shape[1] + offset
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

    snapshot_idxs = np.arange(offset, n_samples, samples_per_tile)
    if clamp_duration is not None:
        clamped_pos = snapshot_idxs // clamp_duration - .5
    snapshots = vis_samples[:, snapshot_idxs].reshape(tile_shape + img_shape)

    # fig, axarr = plt.subplots(*tile_shape, figsize=(tile_shape[1]*2, tile_shape[0]*1.5))
    figsize = (tile_shape[1]*2, tile_shape[0]*1.5)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(*tile_shape, wspace=0, hspace=0)
    # , top=1.-0.5/figsize[1], bottom=0.5/figsize[1],
    #                    left=0.5/figsize[0], right=1-0.5/figsize[0])
    axarr = np.empty(tile_shape, dtype=object)
    for i, snaps_expt in enumerate(snapshots):
        for j, snap in enumerate(snaps_expt):
            axarr[i, j] = plt.subplot(gs[i, j])
            axarr[i, j].tick_params(
                left='off', right='off', bottom='off', top='off',
                labelleft='off', labelbottom='off', labelright='off', labeltop='off')
            axarr[i, j].imshow(snap, interpolation='Nearest', cmap='gray_r')
            if clamp_duration is not None:
                axarr[i, j].axvline(clamped_pos[j], color='C1', lw=1)
                if clamp_window is not None:
                    axarr[i, j].axvline(clamped_pos[j] - clamp_window, color='C1', lw=1)
        if len(titles) > 0:
            print('ticks')
            axarr[i, 0].tick_params(labelleft='on')
            tick_locs = .5*(img_shape[0] + 1)
            plt.yticks(tick_locs, titles[i], rotation='horizontal')
    # fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(os.path.join(make_figure_folder(), savename + '.png'))


def main(config_dict):
    sample_files = config_dict['sample_files']
    data_idx = config_dict['data_idx']
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
    if 'clamp_duration' in config_dict.keys():
        clamp_duration = config_dict['clamp_duration']
    else:
        clamp_duration = None
    if 'clamp_window' in config_dict.keys():
        clamp_window = config_dict['clamp_window']
    else:
        clamp_window = None

    frame_list = []
    n_samples = []
    for i, f in enumerate(sample_files):
        tmp, ns = load_samples(
            f, data_idx, img_shape, n_labels, average=average,
            show_probs=show_probs, rbm=rbm,
            show_hidden=show_hidden, savename=savename + '_{:02d}'.format(i))
        frame_list.append(tmp)
        n_samples.append(ns)
    assert np.all(np.array(n_samples) == n_samples[0])
    n_samples = n_samples[0]

    if make_video:
        plot_animation(frame_list, img_shape, n_samples, titles=titles,
                       show_hidden=show_hidden, savename=savename)
    else:
        tile_shape = tuple(config_dict['tile_shape'])
        try:
            offset = config_dict['offset']
        except KeyError:
            offset = 0
        samples_per_tile = config_dict['samples_per_tile']

        samples = np.array(frame_list, dtype=float).reshape(-1, n_samples, np.prod(img_shape))
        assert tile_shape[0] == len(samples)
        plot_samples(samples, tile_shape, img_shape,
                     samples_per_tile=samples_per_tile, offset=offset,
                     savename=savename, titles=titles)
        # plot_samples(samples, tile_shape, img_shape,
        #              samples_per_tile=samples_per_tile, offset=offset,
        #              savename=savename, titles=titles,
        #              clamp_duration=clamp_duration, clamp_window=clamp_window)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml that specifies'
              ' the configuration.')
        sys.exit()

    # load list of identifiers used for selecting data
    with open(sys.argv[1]) as f:
        config = yaml.load(f)

    main(config)
