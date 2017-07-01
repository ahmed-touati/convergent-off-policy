from utils import get_uniform_policy, policy_evaluation, OffPolicyOperator, estimate_stationary_distribution
from algorithms import off_policy, gradient_off_policy, online_gradient_off_policy
from lib.envs.counterexample import CounterExample
from value_functions import CounterExampleValueFunction

import matplotlib.pyplot as plt
import numpy as np
import pdb
from tqdm import tqdm


if __name__=='__main__':

    env = CounterExample()
    nA = env.nA
    nS = env.nS
    value_function = CounterExampleValueFunction()
    target_policy = np.array([[0, 1], [0, 1]])
    behaviour_policy = get_uniform_policy(nS, nA)
    theta = np.ones(value_function.param_shape)
    theta[1] = 2

    discount_factor = 0.99
    lambda_param = 1
    decay = False
    n_runs = 5

    print 'estimating stationary distribution ..'
    mu = estimate_stationary_distribution(env, behaviour_policy)
    tree_backup_operator = OffPolicyOperator(env, target_policy, behaviour_policy, mu, discount_factor=discount_factor,
                                             lambda_param=lambda_param)
    tree_backup_operator.compute_projection_operator(value_function.PHI)
    tree_backup_operator.compute_operators(type='TB')
    #
    retrace_operator = OffPolicyOperator(env, target_policy, behaviour_policy, mu, discount_factor=discount_factor,
                                         lambda_param=lambda_param)
    retrace_operator.compute_projection_operator(value_function.PHI)
    retrace_operator.compute_operators(type='Retrace')

    trueQ = None
    # to compute MSE error instead of RMSPBE, uncomment the next line
    # trueQ, trueV = policy_evaluation(env, target_policy, discount_factor=discount_factor)

    num_episodes = 100
    errors = np.zeros((n_runs, num_episodes))
    retrace_errors = np.zeros((n_runs, num_episodes))

    print 'running off-policy algo ...'
    for r in range(n_runs):
        errors[r, :] = off_policy(env, value_function, target_policy, behaviour_policy, trueQ, tree_backup_operator,
                                  theta=theta.copy(), discount_factor=discount_factor, lambda_param=lambda_param,
                                  alpha=0.01, num_episodes=num_episodes, decay=decay, type='TB')

        retrace_errors[r, :] = off_policy(env, value_function, target_policy, behaviour_policy, trueQ, retrace_operator,
                                          theta=theta.copy(), discount_factor=discount_factor, lambda_param=lambda_param,
                                          alpha=0.01, num_episodes=num_episodes, decay=decay, type='Retrace')
    errors_mean = np.mean(errors, axis=0)
    errors_std = np.std(errors, axis=0)

    retrace_errors_mean = np.mean(retrace_errors, axis=0)
    retrace_errors_std = np.std(retrace_errors, axis=0)

    print 'save plot ...'
    plt.clf()
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    params = {'text.usetex': True,
              'font.size': 25,
              'font.family': 'lmodern',
              'text.latex.unicode': True,
              }
    plt.rcParams.update(params)

    plt1 = plt.subplot(111, adjustable='box-forced')
    plt1.spines["top"].set_visible(False)
    plt1.spines["right"].set_visible(False)
    plt1.set_xlabel('episode')
    plt1.set_ylabel('RMSBPE')
    plt1.fill_between(np.arange(len(errors_mean)), errors_mean - errors_std, errors_mean + errors_std, alpha=0.1,
                      color='g')
    plt1.plot(np.arange(len(errors_mean)), errors_mean, color="g", label="TB($\lambda$)")

    plt1.fill_between(np.arange(len(retrace_errors_mean)), retrace_errors_mean - retrace_errors_std,
                      retrace_errors_mean + retrace_errors_std, alpha=0.1,
                      color='r')
    plt1.plot(np.arange(len(retrace_errors_mean)), retrace_errors_mean, color="r", label="Retrace($\lambda$)")
    plt1.legend(loc='best')
    plt.savefig('plots/counterexample.png', bbox_inches='tight')

    print 'running gradient off-policy algo ...'
    num_episodes = 100
    errors = np.zeros((n_runs, num_episodes))
    retrace_errors = np.zeros((n_runs, num_episodes))

    for r in range(n_runs):

        errors[r, :] = gradient_off_policy(env, value_function, target_policy, behaviour_policy, trueQ, tree_backup_operator,
                                           theta=theta.copy(), discount_factor=discount_factor, lambda_param=lambda_param,
                                           alpha=0.1, num_episodes=num_episodes, decay=decay, type='TB')

        retrace_errors[r, :] = gradient_off_policy(env, value_function, target_policy, behaviour_policy, trueQ, retrace_operator,
                                                   theta=theta.copy(), discount_factor=discount_factor, lambda_param=lambda_param,
                                                   alpha=0.1, num_episodes=num_episodes, decay=decay, type='Retrace')
    errors_mean = np.mean(errors, axis=0)
    errors_std = np.std(errors, axis=0)

    retrace_errors_mean = np.mean(retrace_errors, axis=0)
    retrace_errors_std = np.std(retrace_errors, axis=0)

    print 'save plot ...'
    plt.clf()
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    params = {'text.usetex': True,
              'font.size': 25,
              'font.family': 'lmodern',
              'text.latex.unicode': True,
              }
    plt.rcParams.update(params)

    plt1 = plt.subplot(111, adjustable='box-forced')
    plt1.spines["top"].set_visible(False)
    plt1.spines["right"].set_visible(False)
    plt1.set_xlabel('episode')
    plt1.set_ylabel('RMSBPE')
    plt1.fill_between(np.arange(len(errors_mean)), errors_mean - errors_std, errors_mean + errors_std, alpha=0.1, color='g')
    plt1.plot(np.arange(len(errors_mean)), errors_mean, color="g", label="GTB($\lambda$)")

    plt1.fill_between(np.arange(len(retrace_errors_mean)), retrace_errors_mean - retrace_errors_std, retrace_errors_mean + retrace_errors_std, alpha=0.1,
                     color='r')
    plt1.plot(np.arange(len(retrace_errors_mean)), retrace_errors_mean, color="r", label="GRetrace($\lambda$)")
    plt1.legend(loc='best')
    plt.savefig('plots/counterexample_gradient.png', bbox_inches='tight')
