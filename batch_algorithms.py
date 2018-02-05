import numpy as np


def off_policy(value_function, target_policy, data, lambda_param, discount_factor, type, num_episodes,
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


def gradient_off_policy(value_function, target_policy, data, lambda_param, discount_factor, return_type, num_episodes,
                              obj_function, alpha_omega_0, alpha_theta_0):
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
            iteration = i * len(episode['states']) + idx

            if i % 20 == 0:
                alpha_omega = alpha_omega_0 / np.sqrt(i + 1)
                alpha_theta = alpha_theta_0 / np.sqrt(i + 1)
            old_omega = omega.copy()
            old_theta = theta.copy()

            phi = value_function.feature(state, action)
            q = np.dot(old_theta, phi)

            if return_type == 'TB':
                kappa = target_prob
            elif return_type == 'Retrace':
                kappa = min([1, target_prob / behavior_prob])
            else:
                raise NotImplementedError()

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

            omega = old_omega + alpha_omega * (delta * e - np.dot(old_omega, phi) * phi)

            theta = old_theta - alpha_theta * np.dot(old_omega, e) * \
                                (discount_factor * expected_phiprime - phi)
        if i > num_episodes:
            break
    return errors


def AB_Trace(value_function, target_policy, behavior_policy, data, lambda_param, discount_factor, num_episodes,
                   obj_function, alpha_omega_0, alpha_theta_0):
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
            iteration = i * len(episode['states']) + idx

            if i % 10 == 0:
                alpha_omega = alpha_omega_0 / np.sqrt(i + 1)
                alpha_theta = alpha_theta_0 / np.sqrt(i + 1)
            old_omega = omega.copy()
            old_theta = theta.copy()

            phi = value_function.feature(state, action)
            q = np.dot(old_theta, phi)

            kappa = min([1, target_prob / behavior_prob])

            e *= discount_factor * lambda_param * kappa
            e += phi

            if idx == len(episode['states']) - 1:
                expected_phiprime = 0
                weighted_phiprime = 0
            else:
                expected_phiprime = np.sum(
                    [target_policy[next_state, a] * value_function.feature(next_state, a)
                     for a in np.arange(nA)], axis=0)
                weighted_phiprime = np.sum(
                    [min(target_policy[next_state, a], behavior_policy[next_state, a]) * value_function.feature(next_state, a)
                     for a in np.arange(nA)], axis=0)

            V = np.dot(old_theta, expected_phiprime)
            td_target = reward + discount_factor * V
            delta = td_target - q

            omega = old_omega + alpha_omega * (delta * e - np.dot(old_omega, phi) * phi)

            theta = old_theta + alpha_theta * (delta * e - discount_factor * np.dot(old_omega, e) *
                                            (expected_phiprime - lambda_param * weighted_phiprime))
        if i > num_episodes:
            break
    return errors


def GQ(value_function, target_policy, data, lambda_param, discount_factor, num_episodes, obj_function,
       alpha_omega_0, alpha_theta_0):
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
            iteration = i * len(episode['states']) + idx

            if i % 10 == 0:
                alpha_omega = alpha_omega_0 / np.sqrt(i + 1)
                alpha_theta = alpha_theta_0 / np.sqrt(i + 1)
            old_omega = omega.copy()
            old_theta = theta.copy()

            phi = value_function.feature(state, action)
            q = np.dot(old_theta, phi)

            kappa = target_prob / behavior_prob

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

            omega = old_omega + alpha_omega * (delta * e - np.dot(old_omega, phi) * phi)

            theta = old_theta + alpha_theta * (delta * e - discount_factor * (1 - lambda_param) * np.dot(old_omega, e) *
                                               expected_phiprime)
        if i > num_episodes:
            break
    return errors