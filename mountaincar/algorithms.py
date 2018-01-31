import numpy as np
from mountain_car import ACTIONS, behavior_policy, takeAction, POSITION_MAX, target_policy, DISCOUNT_FACTOR, \
    ValueFunction, get_behavior_prob, get_target_prob


def off_policy(value_function, data, lambda_param, return_type, obj_function, alpha):

    theta = np.zeros(value_function.maxSize)
    errors = []

    for i, episode in enumerate(data):
        error = obj_function(theta)
        errors.append(error)

        if i % 10 == 0:
            print('episode %d, error %f' % (i, error))
        # initialising eligibility traces
        e = np.zeros(shape=theta.shape)
        for idx, (position, velocity, action, newPosition, newVelocity, reward) in enumerate(
                zip(episode['positions'][:-1],
                    episode['velocities'][:-1],
                    episode['actions'][:-1],
                    episode['positions'][1:],
                    episode['velocities'][1:],
                    episode['rewards'],
                    )):

            old_theta = theta.copy()

            phi = value_function.feature(position, velocity, action)
            q = np.dot(old_theta, phi)

            if return_type == 'TB':
                kappa = get_target_prob(position, velocity, action)
            elif return_type == 'Retrace':
                kappa = min([1, get_target_prob(position, velocity, action) / get_behavior_prob(position, velocity, action)])

            e *= DISCOUNT_FACTOR * lambda_param * kappa
            e += phi

            if newPosition == POSITION_MAX:
                expected_phiprime = 0
            else:
                expected_phiprime = np.sum(
                    [get_target_prob(newPosition, newVelocity, a) * value_function.feature(newPosition, newVelocity, a)
                     for a in ACTIONS], axis=0)

            V = np.dot(old_theta, expected_phiprime)
            td_target = reward + DISCOUNT_FACTOR * V
            delta = td_target - q

            theta = old_theta + alpha * delta * e

        # if i > num_episodes:
        #     break
    return errors


def gradient_off_policy(value_function, data, lambda_param, return_type, obj_function, alpha_omega, alpha_theta):

    theta = np.zeros(value_function.maxSize)
    omega = np.zeros(value_function.maxSize)
    errors = []

    for i, episode in enumerate(data):
        error = obj_function(theta)
        errors.append(error)

        if i % 10 == 0:
            print('episode %d, error %f' % (i, error))
        # initialising eligibility traces
        e = np.zeros(shape=theta.shape)
        for idx, (position, velocity, action, newPosition, newVelocity, reward) in enumerate(
                zip(episode['positions'][:-1],
                    episode['velocities'][:-1],
                    episode['actions'][:-1],
                    episode['positions'][1:],
                    episode['velocities'][1:],
                    episode['rewards'],
                    )):

            old_theta = theta.copy()
            old_omega = omega.copy()

            phi = value_function.feature(position, velocity, action)
            q = np.dot(old_theta, phi)

            if return_type == 'TB':
                kappa = get_target_prob(position, velocity, action)
            elif return_type == 'Retrace':
                kappa = min([1, get_target_prob(position, velocity, action) / get_behavior_prob(position, velocity, action)])

            e *= DISCOUNT_FACTOR * lambda_param * kappa
            e += phi

            if newPosition == POSITION_MAX:
                expected_phiprime = 0
            else:
                expected_phiprime = np.sum(
                    [get_target_prob(newPosition, newVelocity, a) * value_function.feature(newPosition, newVelocity, a)
                     for a in ACTIONS], axis=0)

            V = np.dot(old_theta, expected_phiprime)
            td_target = reward + DISCOUNT_FACTOR * V
            delta = td_target - q

            omega = old_omega + alpha_omega * (delta * e - np.dot(old_omega, phi) * phi)

            theta = old_theta - alpha_theta * np.dot(old_omega, e) * \
                                (DISCOUNT_FACTOR * expected_phiprime - phi)
        # if i > num_episodes:
        #     break
    return errors


def AB_Trace(value_function, data, lambda_param, return_type, obj_function, alpha_omega, alpha_theta):

    theta = np.zeros(value_function.maxSize)
    omega = np.zeros(value_function.maxSize)
    errors = []

    for i, episode in enumerate(data):
        error = obj_function(theta)
        errors.append(error)

        if i % 10 == 0:
            print('episode %d, error %f' % (i, error))
        # initialising eligibility traces
        e = np.zeros(shape=theta.shape)
        for idx, (position, velocity, action, newPosition, newVelocity, reward) in enumerate(
                zip(episode['positions'][:-1],
                    episode['velocities'][:-1],
                    episode['actions'][:-1],
                    episode['positions'][1:],
                    episode['velocities'][1:],
                    episode['rewards'],
                    )):

            old_theta = theta.copy()
            old_omega = omega.copy()

            phi = value_function.feature(position, velocity, action)
            q = np.dot(old_theta, phi)

            if return_type == 'TB':
                kappa = get_target_prob(position, velocity, action)
            elif return_type == 'Retrace':
                kappa = min([1, get_target_prob(position, velocity, action) / get_behavior_prob(position, velocity, action)])

            e *= DISCOUNT_FACTOR * lambda_param * kappa
            e += phi

            if newPosition == POSITION_MAX:
                expected_phiprime = 0
                weighted_phiprime = 0
            else:
                expected_phiprime = np.sum(
                    [get_target_prob(newPosition, newVelocity, a) * value_function.feature(newPosition, newVelocity, a)
                     for a in ACTIONS], axis=0)
                weighted_phiprime = np.sum(
                    [min(1, get_target_prob(newPosition, newVelocity, a) / get_behavior_prob(position, velocity, action))
                     * value_function.feature(newPosition, newVelocity, a)
                     for a in ACTIONS], axis=0)

            V = np.dot(old_theta, expected_phiprime)
            td_target = reward + DISCOUNT_FACTOR * V
            delta = td_target - q

            omega = old_omega + alpha_omega * (delta * e - np.dot(old_omega, phi) * phi)

            theta = old_theta + alpha_theta * (delta * e - DISCOUNT_FACTOR * np.dot(old_omega, e) *
                                               (expected_phiprime - lambda_param * weighted_phiprime))
        # if i > num_episodes:
        #     break
    return errors


def GQ(value_function, data, lambda_param, return_type, obj_function, alpha_omega, alpha_theta):

    theta = np.zeros(value_function.maxSize)
    omega = np.zeros(value_function.maxSize)
    errors = []

    for i, episode in enumerate(data):
        error = obj_function(theta)
        errors.append(error)

        if i % 10 == 0:
            print('episode %d, error %f' % (i, error))
        # initialising eligibility traces
        e = np.zeros(shape=theta.shape)
        for idx, (position, velocity, action, newPosition, newVelocity, reward) in enumerate(
                zip(episode['positions'][:-1],
                    episode['velocities'][:-1],
                    episode['actions'][:-1],
                    episode['positions'][1:],
                    episode['velocities'][1:],
                    episode['rewards'],
                    )):

            old_theta = theta.copy()
            old_omega = omega.copy()

            phi = value_function.feature(position, velocity, action)
            q = np.dot(old_theta, phi)

            if return_type == 'TB':
                kappa = get_target_prob(position, velocity, action)
            elif return_type == 'Retrace':
                kappa = min([1, get_target_prob(position, velocity, action) / get_behavior_prob(position, velocity, action)])

            e *= DISCOUNT_FACTOR * lambda_param * kappa
            e += phi

            if newPosition == POSITION_MAX:
                expected_phiprime = 0
            else:
                expected_phiprime = np.sum(
                    [get_target_prob(newPosition, newVelocity, a) * value_function.feature(newPosition, newVelocity, a)
                     for a in ACTIONS], axis=0)

            V = np.dot(old_theta, expected_phiprime)
            td_target = reward + DISCOUNT_FACTOR * V
            delta = td_target - q

            omega = old_omega + alpha_omega * (delta * e - np.dot(old_omega, phi) * phi)

            theta = old_theta + alpha_theta * (delta * e - DISCOUNT_FACTOR * (1 - lambda_param) * np.dot(old_omega, e) *
                                               expected_phiprime)
        # if i > num_episodes:
        #     break
    return errors



