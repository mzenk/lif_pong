# script that should compute the success rate of an agent given the predictions of the RBM
# under construction...
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
from utils.data_mgmt import get_data_path, make_data_folder
from utils import average_helper


# copied from animate_gibbs
class Pong_agent(object):
    def __init__(self, max_pos, paddle_len=3, max_step=None):
        if max_step is None:
            # maximum distance traveled in one update step
            max_step = 1.
        self.pos_range = [0., max_pos]
        self.paddle_len = paddle_len
        self.max_step = max_step
        self.pos = 0.

    def update_pos(self, prediction):
        # buggy?
        # step = prediction - pos
        # pos += np.clip(step, -self.max_step, self.max_step)
        # alternatively
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
        success = np.mean(dist <= .5 * self.paddle_len)
        return success, dist, paddle_trace

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('No arguments given. Using default.')
        pot_str = 'pong'
        win_size = 48
    else:
        pot_str = sys.argv[1]
        win_size = sys.argv[2]
        # can give 'baseline' as second argument

    img_shape = (36, 48)
    n_labels = 12
    lab_width = img_shape[0] / n_labels
    # ball velocity measured in pixels per time between two clamping frames
    v_ball = 1.

    # load targets and prediction data
    data_file = pot_str + '_var_start{}x{}'.format(*img_shape)
    pred_file = pot_str + '_win{}_prediction'.format(win_size)
    save_file = pot_str + '_win{}_agent_performance'.format(win_size)

    with np.load('../datasets/' + data_file + '.npz') as d:
        _, _, test_set = d[d.keys()[0]]
        tmp = test_set[1]
        tmp[np.all(tmp == 0, axis=1)] = 1.
        targets = np.average(np.tile(np.arange(n_labels), (len(tmp), 1)),
                             weights=tmp, axis=1)
    with np.load(get_data_path('gibbs_sampling') + pred_file + '.npz') as d:
        avg_lab = d['label']
        avg_vis = d['last_col']
        if 'data_idx' in d.keys():
            data_idx = d['data_idx']
        else:
            data_idx = np.arange(len(avg_vis))

    lab2pxl = img_shape[0] / n_labels
    # targets = average_helper(n_labels, test_set[1]) * lab2pxl
    # compare vis_prediction not to label but to actual pixels
    last_col = test_set[0].reshape((-1,) + img_shape)[..., -1]
    targets = average_helper(img_shape[0], last_col)[data_idx]

    lab_prediction = np.zeros(avg_lab.shape[:-1])
    vis_prediction = np.zeros(avg_vis.shape[:-1])
    for i in range(len(avg_lab)):
        lab_prediction[i] = average_helper(n_labels, avg_lab[i])
        vis_prediction[i] = average_helper(img_shape[0], avg_vis[i])
    predictions = vis_prediction.T
    # exclude fully clamped prediction because it is always correct

    # baseline agent: follows y coordinate of ball
    if win_size == 'baseline':
        save_file = 'baseline_agent_performance'
        imgs = test_set[0].reshape((-1,) + img_shape)
        baseline_preds = np.zeros((img_shape[1], len(targets)))
        for i in range(len(baseline_preds)):
            curr_col = imgs[..., i]
            baseline_preds[i] = average_helper(img_shape[0], curr_col)
        # print((np.abs(baseline_preds[-1] - targets) < .5 * lab_width).mean())
        predictions = baseline_preds

    # exclude fully clamped prediction because it is always correct?
    predictions = predictions[:-1]
    # sensible max_step range: one paddle length per one ball x-step => v=1.
    speed_range = np.linspace(0, 1.2 * v_ball, 100)
    success_rates = np.zeros_like(speed_range)
    distances = np.zeros((len(speed_range), len(targets)))
    n_recorded = 1000
    traces = np.zeros((speed_range.shape[0], predictions.shape[0], n_recorded))
    for i, speed in enumerate(speed_range):
        print('sweep agent speed ({} of {})...'
              ''.format(i + 1, len(speed_range)))
        my_agent = Pong_agent(img_shape[0], paddle_len=lab_width,
                              max_step=speed)
        success_rates[i], distances[i], tmp = \
            my_agent.simulate_games(predictions, targets)
        traces[i] = tmp[:, :n_recorded]

    # save data - speed normalized to ball speed:
    np.savez_compressed(make_data_folder(__file__) + save_file,
                        successes=success_rates, distances=distances,
                        traces=traces, speeds=speed_range/v_ball)

    # # test
    # p1 = np.sin(np.linspace(0, 10, 100))
    # p1 = np.hstack((p1, np.repeat(p1[-1], 100)))
    # p2 = np.cos(np.linspace(0, 10, 100))
    # p2 = np.hstack((p2, np.repeat(p2[-1], 100)))
    # p = np.vstack((p1, p2)).T
    # t = p[-1]
    # v_agent = .05
    # my_agent = Pong_agent(1., .1, v_agent)
    # _, _, trace = my_agent.simulate_games(p, t)

    # trace = trace[:, 1]
    # p = p2
    # t = t[1]
    # plt.plot(trace, label='agent')
    # plt.plot(p, label='prediction')
    # plt.plot([0, len(p) - 1], [t, t])
    # plt.legend()
    # plt.savefig('figures/test.png')
