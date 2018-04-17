#! /usr/bin/env python
# script that should compute the success rate of an agent given the predictions of the RBM
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import yaml
from lif_pong.utils.data_mgmt import get_data_path, make_data_folder, load_images
from lif_pong.utils import average_helper


# copied from animate_gibbs
class Pong_agent(object):
    def __init__(self, max_pos, paddle_len=3, max_step=None):
        if max_step is None:
            # maximum distance traveled in one update step
            max_step = 1.
        self.pos_range = [0., max_pos]
        self.paddle_len = paddle_len
        self.max_step = max_step
        self.pos = .5*max_pos

    def update_pos(self, prediction):
        step = np.minimum(self.max_step, np.abs(prediction - self.pos))
        self.pos += np.sign(prediction - self.pos) * step

    def simulate_games(self, predictions, targets):
        """
        arguments:
            predictions: (n_pxls_x, n_instances)
            prediction for each ball position (except last column) and instance
            targets: groundtruth
        returns:
            success: scalar
            dist: (n_instances) distance from final position to target
            paddle_trace: (n_pxls_x, n_instances) all agent positions
        """
        self.pos = .5*np.sum(self.pos_range) * np.ones(predictions.shape[1])

        paddle_trace = np.zeros_like(predictions)
        for i, pred in enumerate(predictions):
            self.update_pos(pred)
            paddle_trace[i] = self.pos.copy()

        dist = np.abs(self.pos - targets)
        success = np.sum(dist <= .5 * self.paddle_len)
        success_std = np.std(dist <= .5 * self.paddle_len) * len(dist)
        return success, success_std, dist, paddle_trace


def test():
    import matplotlib.pyplot as plt
    p1 = np.sin(np.linspace(0, 10, 100))
    p1 = np.hstack((p1, np.repeat(p1[-1], 100)))
    p2 = np.cos(np.linspace(0, 10, 100))
    p2 = np.hstack((p2, np.repeat(p2[-1], 100)))
    p = np.vstack((p1, p2)).T
    t = p[-1]
    v_agent = .05
    my_agent = Pong_agent(1., .1, v_agent)
    _, _, trace = my_agent.simulate_games(p, t)

    trace = trace[:, 1]
    p = p2
    t = t[1]
    plt.plot(trace, label='agent')
    plt.plot(p, label='prediction')
    plt.plot([0, len(p) - 1], [t, t])
    plt.legend()
    plt.savefig('figures/test.png')


def compute_performance(predictions, targets, data_idx, max_speedup=2.,
                        use_labels=False, paddle_width=4, leave_uncovered=1):
    # predictions must have shape (n_instances, n_steps, n_pos)
    # targets must either have same shape as predictions or (n_instances, n_pos)
    n_instances, n_frames, n_pos = predictions.shape
    if targets.size != predictions.size:
        target_pos = average_helper(n_pos, targets)
    else:
        target_pos = np.zeros((n_instances, n_frames))
    predicted_pos = np.zeros((n_instances, n_frames))
    for i in range(n_instances):
        predicted_pos[i] = average_helper(n_pos, predictions[i])
        if targets.size == predictions.size:
            target_pos[i] = average_helper(n_pos, targets[i])

    # for knowing how the agent performs at infinite speed just return the
    # number of cases where the prediction error is smaller than the
    # paddle width. successes has n_frames entries (basically time)
    if max_speedup == np.inf:
        if len(target_pos.shape) == 1:
            target_pos = target_pos.reshape((-1, 1))
        pred_error = np.abs(predicted_pos - target_pos)
        successes = np.sum(pred_error < .5*paddle_width, axis=0)
        # for inspecting failed cases
        wrong_idx = data_idx[np.where(
            pred_error[:, -(leave_uncovered + 1)] > .5*paddle_width)]
        result = {
            'successes': successes,
            'distances': pred_error,  # for infinite speed equivalent
            'speeds': np.inf,
            'n_instances': n_instances,
            'wrong_idx': wrong_idx
        }
        return result

    predicted_pos = predicted_pos.T
    if target_pos.size == predicted_pos.size:
        target_pos = target_pos.T
    # exclude fully clamped prediction because it is always correct
    if leave_uncovered > 0:
        predicted_pos[-leave_uncovered:] = predicted_pos[-leave_uncovered - 1]

    # sensible max_step range: one paddle length per one ball x-step => v=1.
    # ball velocity measured in pixels per time between two clamping frames
    v_ball = 1.
    speed_range = np.linspace(0, max_speedup * v_ball, 100)
    successes = np.zeros_like(speed_range)
    successes_std = np.zeros_like(speed_range)
    distances = np.zeros((len(speed_range), n_instances))
    n_recorded = n_instances//100
    traces = np.zeros((speed_range.shape[0], predicted_pos.shape[0], n_recorded))
    print('sweep agent speed ({} to {})...'.format(speed_range[0], speed_range[-1]))
    for i, speed in enumerate(speed_range):
        my_agent = Pong_agent(n_pos, paddle_len=paddle_width, max_step=speed)
        successes[i], successes_std[i], distances[i], tmp = \
            my_agent.simulate_games(predicted_pos, target_pos)
        traces[i] = tmp[:, :n_recorded]

    # save data - speed normalized to ball speed
    # caution: success_std is n_instances * standard dev. of success rate!
    result = {
        'successes': successes,
        'successes_std': successes_std,
        'distances': distances,
        'traces': traces,
        'speeds': speed_range/v_ball,
        'n_instances': n_instances
    }
    return result


def compute_baseline_performance(img_shape, test_set, data_idx, max_speedup=2.,
                                 paddle_width=3, leave_uncovered=1):
    n_pos = img_shape[0]

    # compute baseline prediction
    imgs = test_set[0].reshape((-1,) + img_shape)[data_idx]
    bl_preds = np.zeros((len(imgs), 1 + img_shape[1], img_shape[0]))
    # for no clamped column insert guess in the centre
    bl_preds[:, 0] = np.ones((bl_preds.shape[0], bl_preds.shape[2]))/n_pos
    bl_preds[:, 1:] = np.swapaxes(imgs, 1, 2)

    # has no knick adaption
    targets = imgs[..., -1]

    # quick and dirty save
    np.savez('prediction', last_col=bl_preds, data_idx=data_idx)
    return compute_performance(bl_preds, targets, data_idx,
                               max_speedup=max_speedup, 
                               paddle_width=paddle_width,
                               leave_uncovered=leave_uncovered)


if __name__ == '__main__':
    # this is 
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml-config file.')
        sys.exit()
    with open(sys.argv[1]) as configfile:
        config = yaml.load(configfile)

    general_dict = config.pop('general')
    agent_kwargs = config.pop('agent_kwargs')
    img_shape = tuple(general_dict['img_shape'])
    _, _, test_set = load_images(general_dict['data_name'])
    start = general_dict['start_idx']
    end = start + general_dict['chunksize']
    data_idx = np.arange(start, end)

    result = compute_baseline_performance(img_shape, test_set, data_idx,
                                          **agent_kwargs)
    np.savez('agent_performance', **result)
    
    # ======== keep for reference ========
    # # compute_performance(...) is a simplified version hereof
    # if len(sys.argv) < 4:
    #     print('Wrong no. of arguments given.')
    #     sys.exit()
    # else:
    #     method = sys.argv[1]
    #     if method == 'lif':
    #         script_name = 'lif_clamp_window'
    #     elif method == 'gibbs':
    #         script_name = 'gibbs_sampling'
    #     else:
    #         print('Sampling method not recognized. Possible: lif, gibbs')
    #         sys.exit()
    #     pot_str = sys.argv[2]
    #     win_size = sys.argv[3]
    #     # can give 'baseline' as second argument
    #     if len(sys.argv) == 5:
    #         name_mod = '_' + sys.argv[4]
    #     else:
    #         name_mod = ''

    # img_shape = (36, 48)
    # n_labels = img_shape[0] // 3
    # lab_width = img_shape[0] / n_labels
    # # ball velocity measured in pixels per time between two clamping frames
    # v_ball = 1.

    # # load targets and prediction data
    # data_file = pot_str + '_var_start{}x{}'.format(*img_shape)
    # pred_path = get_data_path(script_name)
    # pred_file = pot_str + '_win{}{}_prediction'.format(win_size, name_mod)
    # save_file = '{}_{}_win{}{}_agent_performance'.format(
    #     method, pot_str, win_size, name_mod)

    # _, _, test_set = load_images(data_file)
    # tmp = test_set[1]
    # tmp[np.all(tmp == 0, axis=1)] = 1.
    # targets = np.average(np.tile(np.arange(n_labels), (len(tmp), 1)),
    #                      weights=tmp, axis=1)
    # with np.load(pred_path + pred_file + '.npz') as d:
    #     avg_lab = d['label']
    #     avg_vis = d['last_col']
    #     if 'data_idx' in d.keys():
    #         data_idx = d['data_idx']
    #     else:
    #         data_idx = np.arange(len(avg_vis))

    # lab2pxl = img_shape[0] / n_labels
    # # targets = average_helper(n_labels, test_set[1]) * lab2pxl
    # # compare vis_prediction not to label but to actual pixels
    # last_col = test_set[0].reshape((-1,) + img_shape)[..., -1]
    # targets = average_helper(img_shape[0], last_col)[data_idx]

    # lab_prediction = np.zeros(avg_lab.shape[:-1])
    # vis_prediction = np.zeros(avg_vis.shape[:-1])
    # for i in range(len(avg_lab)):
    #     lab_prediction[i] = average_helper(n_labels, avg_lab[i])
    #     vis_prediction[i] = average_helper(img_shape[0], avg_vis[i])
    # predictions = vis_prediction.T
    # # exclude fully clamped prediction because it is always correct

    # # baseline agent: follows y coordinate of ball
    # if win_size == 'baseline':
    #     save_file = 'baseline_agent_performance'
    #     imgs = test_set[0].reshape((-1,) + img_shape)
    #     baseline_preds = np.zeros((img_shape[1], len(targets)))
    #     for i in range(len(baseline_preds)):
    #         curr_col = imgs[..., i]
    #         baseline_preds[i] = average_helper(img_shape[0], curr_col)
    #     # print((np.abs(baseline_preds[-1] - targets) < .5 * lab_width).mean())
    #     predictions = baseline_preds

    # print(len(targets), predictions.shape)
    # # exclude fully clamped prediction because it is always correct?
    # predictions = predictions[:-1]
    # # sensible max_step range: one paddle length per one ball x-step => v=1.
    # speed_range = np.linspace(0, 1.2 * v_ball, 100)
    # successes = np.zeros_like(speed_range)
    # distances = np.zeros((len(speed_range), len(targets)))
    # n_recorded = len(targets)//100
    # traces = np.zeros((speed_range.shape[0], predictions.shape[0], n_recorded))
    # for i, speed in enumerate(speed_range):
    #     print('sweep agent speed ({} of {})...'
    #           ''.format(i + 1, len(speed_range)))
    #     my_agent = Pong_agent(img_shape[0], paddle_len=lab_width,
    #                           max_step=speed)
    #     successes[i], _, distances[i], tmp = \
    #         my_agent.simulate_games(predictions, targets)
    #     traces[i] = tmp[:, :n_recorded]
    # success_rates = successes/len(targets)

    # # save data - speed normalized to ball speed:
    # np.savez_compressed(make_data_folder() + save_file,
    #                     successes=success_rates, distances=distances,
    #                     traces=traces, speeds=speed_range/v_ball)
    # ========

    # # for the "lastcol" plot
    # notshown = 0
    # ydata = []
    # xdata = np.arange(0, 49)
    # for notshown in range(0, 49):
    #     save_file = 'pong_win48_100samples_{}shown'.format(48 - notshown)

    #     _, _, test_set = load_images(data_file)
    #     tmp = test_set[1]
    #     tmp[np.all(tmp == 0, axis=1)] = 1.
    #     targets = np.average(np.tile(np.arange(n_labels), (len(tmp), 1)),
    #                          weights=tmp, axis=1)
    #     with np.load(pred_path + pred_file + '.npz') as d:
    #         avg_lab = d['label']
    #         avg_vis = d['last_col']
    #         if 'data_idx' in d.keys():
    #             data_idx = d['data_idx']
    #         else:
    #             data_idx = np.arange(len(avg_vis))

    #     lab2pxl = img_shape[0] / n_labels
    #     # targets = average_helper(n_labels, test_set[1]) * lab2pxl
    #     # compare vis_prediction not to label but to actual pixels
    #     last_col = test_set[0].reshape((-1,) + img_shape)[..., -1]
    #     targets = average_helper(img_shape[0], last_col)[data_idx]

    #     lab_prediction = np.zeros(avg_lab.shape[:-1])
    #     vis_prediction = np.zeros(avg_vis.shape[:-1])
    #     for i in range(len(avg_lab)):
    #         lab_prediction[i] = average_helper(n_labels, avg_lab[i])
    #         vis_prediction[i] = average_helper(img_shape[0], avg_vis[i])
    #     predictions = vis_prediction.T
    #     # exclude fully clamped prediction because it is always correct

    #     # baseline agent: follows y coordinate of ball
    #     if win_size == 'baseline':
    #         save_file = 'baseline_agent_performance'
    #         imgs = test_set[0].reshape((-1,) + img_shape)
    #         baseline_preds = np.zeros((img_shape[1], len(targets)))
    #         for i in range(len(baseline_preds)):
    #             curr_col = imgs[..., i]
    #             baseline_preds[i] = average_helper(img_shape[0], curr_col)
    #         # print((np.abs(baseline_preds[-1] - targets) < .5 * lab_width).mean())
    #         predictions = baseline_preds

    #     print(len(targets), predictions.shape)
    #     # exclude fully clamped prediction because it is always correct?
    #     if notshown != 0:
    #         predictions = predictions[:-notshown]
    #     # sensible max_step range: one paddle length per one ball x-step => v=1.
    #     speed_range = np.linspace(0, 3. * v_ball, 100)
    #     successes = np.zeros_like(speed_range)
    #     distances = np.zeros((len(speed_range), len(targets)))
    #     n_recorded = len(targets)//100
    #     traces = np.zeros((speed_range.shape[0], predictions.shape[0], n_recorded))
    #     print(notshown)
    #     for i, speed in enumerate(speed_range):
    #         # print('sweep agent speed ({} of {})...'
    #         #       ''.format(i + 1, len(speed_range)))
    #         my_agent = Pong_agent(img_shape[0], paddle_len=lab_width,
    #                               max_step=speed)
    #         successes[i], _, distances[i], tmp = \
    #             my_agent.simulate_games(predictions, targets)
    #         traces[i] = tmp[:, :n_recorded]
    #     success_rates = successes/len(targets)

    #     ydata.insert(0, success_rates.max())

    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    # plt.plot(xdata, ydata, 'o')
    # plt.xlabel('Last shown pixel column index')
    # plt.ylabel('Asymptotic success')
    # plt.savefig('lastshowncol.pdf')
