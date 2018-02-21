from utils import collect_dataset, estimate_true_q, estimate_key_quantities, compute_EM_MSBPE
from algorithms import off_policy, gradient_off_policy, GQ, AB_Trace, extragradient_off_policy
from mountain_car import ValueFunction
import numpy as np
import os
import json

return_type = 'TB'
nepisodes = 2000
lambda_range = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
# lambda_range = [0.5, 0.625, 0.75, 0.875, 1.0]
# alpha_range = [0.005, 0.01, 0.05]
# alpha_range = [0.001, 0.005, 0.01]
alpha_range = [0.01, 0.05, 0.1]

datadir = 'data'
logdir = 'logsGQ'
storedir = 'resultsGQ'
if not os.path.exists(logdir):
    os.mkdir(logdir)
if not os.path.exists(storedir):
    os.mkdir(storedir)

if not os.path.exists(datadir):
    os.mkdir(datadir)
    print('collecting data ...')
    collect_dataset(datadir)
    print('estimating true Q function ...')
    with open(os.path.join(datadir, 'test_points.json'), 'r') as f:
        test_points = json.load(f)
    estimate_true_q(test_points, datadir, nepisodes=1000)

with open(os.path.join(datadir, 'test_points.json'), 'r') as f:
    test_points = json.load(f)
data = np.load(os.path.join(datadir, 'dataset_0.npy'))
true_q = np.load(os.path.join(datadir, 'true_q.npy'))
value_function = ValueFunction()

for lambda_param in lambda_range:
    if not os.path.exists(os.path.join(datadir, 'A-{}.npy'.format(lambda_param))):
        print('estimating key quantities ...')
        A, b, M_inv = estimate_key_quantities(value_function, data, lambda_param, return_type)
        print('saving key quantities ...')
        np.save(os.path.join(datadir, 'A-{}-{}.npy'.format(return_type, lambda_param)), A)
        np.save(os.path.join(datadir, 'b-{}-{}.npy'.format(return_type, lambda_param)), b)
        np.save(os.path.join(datadir, 'M-inv-{}-{}.npy'.format(return_type, lambda_param)), M_inv)
    else:
        A = np.load(os.path.join(datadir, 'A-{}.npy'.format(lambda_param)))
        b = np.load(os.path.join(datadir, 'b-{}.npy'.format(lambda_param)))
        M_inv = np.load(os.path.join(datadir, 'M-inv-{}.npy'.format(lambda_param)))

    MSBPE_function = lambda theta: compute_EM_MSBPE(theta, A, b, M_inv)
    MSE_function = lambda theta: value_function.compute_MSE(theta, test_points, true_q)
    for alpha_omega in alpha_range:
        for alpha_theta in alpha_range:
            GQ_errors = GQ(value_function, data, lambda_param, MSE_function, alpha_omega, alpha_theta,
                           nepisodes, logdir, storedir)

            # AB_MSE_errors, AB_MSBPE_errors = AB_Trace(value_function, data, lambda_param, return_type,
            #                                           MSE_function, MSBPE_function, alpha_omega, alpha_theta,
            #                                           nepisodes, logdir, storedir)
            # gradient_MSE_errors, gradient_MSBPE_errors = gradient_off_policy(value_function, data, lambda_param,
            #                                                                  return_type, MSE_function, MSBPE_function,
            #                                                                  alpha_omega, alpha_theta, nepisodes,
            #                                                                  logdir, storedir)









