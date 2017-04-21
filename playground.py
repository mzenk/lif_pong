import numpy as np
import matplotlib.pyplot as plt
import time, gzip

# Load Pong data
f = gzip.open('pong_fixed_start2.npy.gz', 'rb')
_, _, data = np.load(f)
f.close()

test = data[0][0].reshape((42, 42))
test_label = data[1][0]
padded_label = np.tile(np.expand_dims(test_label, 1),
                       (1, 42/14)).reshape(42, 1)
img_w_label = np.concatenate((test, padded_label), axis=1)
plt.figure()
plt.imshow(img_w_label, interpolation='Nearest', cmap='gray')
plt.savefig('lab.png')

# from generate_data import Trajectory, Const_trajectory

# start = np.array([0., 5])
# my_trajectory = Const_trajectory((12,10), 1., start, 30., 1., np.array([0,0]))

# # test method by adding individual points, e.g.
# points = np.array([[4.5,0],
#                    [4.9,6.3],
#                    [1.3,9.5]])
