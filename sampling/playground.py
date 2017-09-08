import numpy as np
from utils.data_mgmt import get_data_path

sample_file = 'pong_classif_50samples'
with np.load(get_data_path('lif_classification') + sample_file + '.npz') as d:
    lab_samples = d['samples']
