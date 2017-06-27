import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['font.size'] = 14

img_shape = (36, 48)
win_size = 100
n_sampl = 100
fname = 'gauss_uncover{}w{}s'.format(win_size, n_sampl)
with np.load('figures/' + fname + '.npz') as d:
    correct_predictions, distances, dist_std, \
        img_diff, img_diff_std = d[d.keys()[0]]

fractions = np.linspace(0, 1, 20)
# plotting...
if win_size < img_shape[1]:
    xlabel = 'Window position'
else:
    xlabel = 'Uncovered fraction'
plt.figure()
# plt.subplot(121)
plt.errorbar(fractions, distances, fmt='ro', yerr=dist_std)
plt.ylabel('Distance to correct label')
# plt.ylim([0, 3])
plt.xlabel(xlabel)
plt.twinx()
plt.plot(fractions, correct_predictions, 'bo')
plt.ylabel('Correct predictions')
# plt.ylim([0, 1])
plt.gca().spines['right'].set_color('blue')
plt.gca().spines['left'].set_color('red')
plt.title('#samples: {}, window size: {}'.format(n_sampl, win_size))

# plt.subplot(122)
# plt.errorbar(fractions, img_diff, fmt='ro', yerr=img_diff_std)
# plt.ylabel('L2 image dissimilarity')
# plt.xlabel(xlabel)
# plt.tight_layout()
plt.savefig('figures/' + fname + '.pdf'.format(win_size, n_sampl))
