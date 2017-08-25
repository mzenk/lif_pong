# script that should compute the success rate of an agent given the predictions of the RBM
# under construction...
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import sys

sampling_interval = 1.


# copied from animate_gibbs
class Pong_agent(object):
    def __init__(self, n_labels, paddle_len=None, max_step=None):
        if paddle_len is None:
            paddle_len = img_shape[0] / n_labels
        if max_step is None:
            # maximum distance traveled in one update step. For now: double the
            # horizontal ball speed
            max_step = 1.
        self.n_labels = n_labels
        self.paddle_len = paddle_len
        self.max_step = max_step

    def simulate_games(self, predictions, targets):
        """
        arguments:
            predictions: (n_pxls_x - 1, n_instances)
            prediction for each ball position (except last column) and instance
            targets: groundtruth
        returns:
            success: scalar
            dist: (n_instances) distance from final position to target
            paddle_trace: (n_pxls_x - 1, n_instances) all agent positions
        """
        if len(targets.shape) < 2:
            paddle_pos = .5 * self.n_labels
        else:
            paddle_pos = .5 * self.n_labels * np.ones(len(targets))

        paddle_trace = np.zeros_like(predictions)
        for i, pred in enumerate(predictions):
            # buggy
            # step = pred - paddle_pos
            # paddle_pos += np.clip(step, -self.max_step, self.max_step)
            # alternatively: set to prediction if < max_step else max_step
            step = np.minimum(self.max_step, np.abs(pred - paddle_pos))
            paddle_pos += np.sign(pred - paddle_pos) * step
            paddle_trace[i] = paddle_pos

        success = np.mean(np.abs(paddle_pos - targets) <= .5)  # or .5*paddle_len?
        dist = np.abs(paddle_pos - targets)
        return success, dist, paddle_trace


img_shape = (36, 48)
pot_str = sys.argv[1]
win_size = sys.argv[2]
n_labels = 12
lab_width = img_shape[0] / n_labels
# ball velocity measured on label scale
v_ball = 1. / lab_width

# load targets and prediction data
data_file = pot_str + '_var_start{}x{}'.format(*img_shape)
pred_file = pot_str[:4] + '_pred{}w'.format(win_size)
save_file = '{}w{}agent_performance'.format(pot_str, win_size)
with np.load('../datasets/' + data_file + '.npz') as d:
    _, _, test_set = d[d.keys()[0]]
    tmp = test_set[1]
    tmp[np.all(tmp == 0, axis=1)] = 1.
    targets = np.average(np.tile(np.arange(n_labels), (len(tmp), 1)),
                         weights=tmp, axis=1)
with np.load('data/' + pred_file + '.npz') as d:
    predictions = d['pred'][:img_shape[1]]

# # baseline agent: follows y coordinate of ball
# save_file = 'baseline'
# imgs = test_set[0].reshape((-1,) + img_shape)
# baseline_preds = np.zeros((img_shape[1], len(targets)))
# for i, p in enumerate(baseline_preds):
#     curr_col = imgs[..., i]
#     p = np.average(np.tile(np.arange(img_shape[0]), (len(curr_col), 1)),
#                    weights=curr_col, axis=1) / lab_width
# print((np.abs(baseline_preds[-1] - targets) > .5).sum())

# sensible max_step range: one paddle length per one ball x-step => v=1.
speed_range = np.linspace(0, v_ball, 100)
success_rates = np.zeros_like(speed_range)
distances = np.zeros((len(speed_range), len(targets)))
n_recorded = 1000
traces = np.zeros((speed_range.shape[0], predictions.shape[0], n_recorded))
for i, speed in enumerate(speed_range):
    print('sweep agent speed ({} of {})...'.format(i + 1, len(speed_range)))
    my_agent = Pong_agent(n_labels, paddle_len=lab_width, max_step=speed)
    success_rates[i], distances[i], tmp = \
        my_agent.simulate_games(predictions, targets)
    traces[i] = tmp[:, :n_recorded]

# save data - speed normalized to ball speed:
np.savez_compressed('data/' + save_file,
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
