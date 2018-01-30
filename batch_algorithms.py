import numpy as np


def batch_off_policy(value_function, target_policy, data, lambda_param, discount_factor, type, num_episodes,
                     obj_function, alpha):
    nS = value_function.nS
    nA = value_function.nA

    theta = np.zeros(value_function.param_shape)
    errors = []

    for i, episode in enumerate(data):
        error = obj_function(theta)
        errors.append(error)

        if i % 10 == 0:
            print('episode %d, error %f' % (i, error))
        # initialising eligibility traces
        e = np.zeros(shape=theta.shape)
        for idx, (state, action, next_state, reward, behavior_prob, target_prob) in enumerate(
                zip(episode['states'][:-1],
                    episode['actions'],
                    episode['states'][1:],
                    episode['rewards'],
                    episode['behavior_probs'],
                    episode['target_probs'])):
            old_theta = theta.copy()

            phi = value_function.feature(state, action)
            q = np.dot(old_theta, phi)

            if type == 'TB':
                kappa = target_prob
            elif type == 'Retrace':
                kappa = min([1, target_prob / behavior_prob])

            e *= discount_factor * lambda_param * kappa
            e += phi

            if idx == len(episode['states']) - 1:
                expected_phiprime = 0
            else:
                expected_phiprime = np.sum(
                    [target_policy[next_state, a] * value_function.feature(next_state, a)
                     for a in np.arange(nA)], axis=0)

            V = np.dot(old_theta, expected_phiprime)
            td_target = reward + discount_factor * V
            delta = td_target - q

            theta += alpha * delta * e

        if i > num_episodes:
            break
    return errors


def batch_gradient_off_policy(value_function, target_policy, data, lambda_param, discount_factor, type, num_episodes,
                              obj_function, alpha):
    nS = value_function.nS
    nA = value_function.nA

    theta = np.zeros(value_function.param_shape)
    omega = np.zeros(theta.shape)
    errors = []

    for i, episode in enumerate(data):
        error = obj_function(theta)
        errors.append(error)

        if i % 10 == 0:
            print('episode %d, error %f' % (i, error))
        # initialising eligibility traces
        e = np.zeros(shape=theta.shape)
        for idx, (state, action, next_state, reward, behavior_prob, target_prob) in enumerate(
                zip(episode['states'][:-1],
                    episode['actions'],
                    episode['states'][1:],
                    episode['rewards'],
                    episode['behavior_probs'],
                    episode['target_probs'])):
            old_omega = omega.copy()
            old_theta = theta.copy()

            phi = value_function.feature(state, action)
            q = np.dot(old_theta, phi)

            if type == 'TB':
                kappa = target_prob
            elif type == 'Retrace':
                kappa = min([1, target_prob / behavior_prob])

            e *= discount_factor * lambda_param * kappa
            e += phi

            if idx == len(episode['states']) - 1:
                expected_phiprime = 0
            else:
                expected_phiprime = np.sum(
                    [target_policy[next_state, a] * value_function.feature(next_state, a)
                     for a in np.arange(nA)], axis=0)

            V = np.dot(old_theta, expected_phiprime)
            td_target = reward + discount_factor * V
            delta = td_target - q

            omega = old_omega + alpha * (delta * e - np.dot(old_omega, phi) * phi)

            theta = old_theta - alpha * np.dot(old_omega, e) * \
                                (discount_factor * expected_phiprime - phi)
        if i > num_episodes:
            break
    return errors
