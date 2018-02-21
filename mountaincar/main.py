from utils import collect_dataset, estimate_true_q, estimate_key_quantities, compute_EM_MSBPE
from algorithms import off_policy, gradient_off_policy, GQ, AB_Trace, extragradient_off_policy
from mountain_car import ValueFunction
import numpy as np
import os
import json
import matplotlib.pyplot as plt

if __name__=='__main__':

    nepisodes = 500
    lambda_param = 1
    return_type = 'Retrace'

    datadir = 'data'
    if not os.path.exists(datadir):
        os.mkdir(datadir)
        print('collecting data ...')
        collect_dataset(datadir)
        print('estimating true Q function ...')
        with open(os.path.join(datadir, 'test_points.json'), 'r') as f:
            test_points = json.load(f)
        estimate_true_q(test_points, datadir, nepisodes=100)

    with open(os.path.join(datadir, 'test_points.json'), 'r') as f:
        test_points = json.load(f)
    data = np.load(os.path.join(datadir, 'dataset_0.npy'))
    true_q = np.load(os.path.join(datadir, 'true_q.npy'))
    value_function = ValueFunction()

    print('estimating key quantities ...')
    test_data = np.load(os.path.join(datadir, 'dataset_0.npy'))
    A, b, M_inv = estimate_key_quantities(value_function, test_data, lambda_param, return_type)

    MSBPE_function = lambda theta: compute_EM_MSBPE(theta, A, b, M_inv)
    MSE_function = lambda theta: value_function.compute_MSE(theta, test_points, true_q)

    # import pdb
    # pdb.set_trace()

    alpha_omega = 0.01
    alpha_theta = 0.001

    #GQ_errors = GQ(value_function, data, lambda_param, MSE_function, alpha_omega, alpha_theta, nepisodes)

    alpha_omega = 0.01
    alpha_theta = 0.001

    AB_MSE_errors, AB_MSBPE_errors = AB_Trace(value_function, data, lambda_param, return_type, MSE_function, MSBPE_function,
                         alpha_omega, alpha_theta, nepisodes)

    gradient_MSE_errors, gradient_MSBPE_errors = gradient_off_policy(value_function, data, lambda_param, return_type, MSE_function, MSBPE_function,
                                          alpha_omega, alpha_theta, nepisodes)
    # extra_erros = extragradient_off_policy(value_function, data, lambda_param, return_type, obj_function,
    #                                        alpha_omega, alpha_theta, nepisodes)

    #plt.plot(np.arange(len(GQ_errors)), GQ_errors, label='GQ')
    plt.plot(np.arange(len(AB_MSBPE_errors)), AB_MSBPE_errors, label='AB-Trace')
    plt.plot(np.arange(len(gradient_MSBPE_errors)), gradient_MSBPE_errors, label='gradient off policy')
    plt.legend()
    plt.show()


