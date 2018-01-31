from utils import collect_dataset, estimate_true_q
import numpy as np
import os

if __name__=='__main__':
    logdir = 'data'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        print('collecting data ...')
        collect_dataset(logdir)
        print('estimating true Q function ...')
        test_points = np.load(os.path.join(logdir, 'test_points.npy'))
        estimate_true_q(test_points, logdir, nepisodes=100)


