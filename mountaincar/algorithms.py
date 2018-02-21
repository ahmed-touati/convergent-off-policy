import numpy as np
from mountain_car import ACTIONS, behavior_policy, takeAction, POSITION_MAX, target_policy, DISCOUNT_FACTOR, \
    ValueFunction, get_behavior_prob, get_target_prob
from tensorboardX import SummaryWriter
import os
import json

def off_policy(value_function, data, lambda_param, return_type, obj_function, alpha, nepisodes):

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

        if i > nepisodes:
            break
    return errors


def gradient_off_policy(value_function, data, lambda_param, return_type, MSE_function, MSBPE_function,
                        alpha_omega_0, alpha_theta_0, nepisodes, logdir, storedir):

    logfile = os.path.join(logdir, '{}-{}-{}-{}-{}'.format('gradient', return_type, lambda_param, alpha_omega_0, alpha_theta_0))
    storefile = os.path.join(storedir, '{}-{}-{}-{}-{}'.format('gradient', return_type, lambda_param, alpha_omega_0, alpha_theta_0))
    # Log
    log = SummaryWriter(logfile)
    print('Writing logs to {}'.format(logfile))

    theta = np.zeros(value_function.maxSize)
    omega = np.zeros(value_function.maxSize)
    MSE_errors = []
    MSBPE_errors = []
    for i, episode in enumerate(data):
        MSE_error = MSE_function(theta)
        MSBPE_error = MSBPE_function(theta)
        MSE_errors.append(MSE_error)
        MSBPE_errors.append(MSBPE_error)

        log.add_scalar('MSE error', MSE_error, i)
        log.add_scalar('MSBPE error', MSBPE_error, i)

        # if i % 10 == 0:
        #     print('episode %d, MSE %f, MSBPE %f' % (i, MSE_error, MSBPE_error))

        if (i > nepisodes) or (MSBPE_error > 50):
            break
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
            iteration = i * len(episode['rewards']) + idx

            # if i > 1000:
            #     alpha_omega = alpha_omega_0 / np.sqrt(i)
            #     alpha_theta = alpha_theta_0 / np.sqrt(i)
            # else:
            alpha_omega = alpha_omega_0
            alpha_theta = alpha_theta_0

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

    print('saving errors in {}'.format(storefile))
    errors = {
        'MSE errors': MSE_errors,
        'MSBPE errors': MSBPE_errors
    }
    with open(storefile, 'w') as f:
        json.dump(errors, f)
    return MSE_errors, MSBPE_errors


def AB_Trace(value_function, data, lambda_param, return_type, MSE_function, MSBPE_function,
             alpha_omega_0, alpha_theta_0, nepisodes, logdir, storedir):

    logfile = os.path.join(logdir, '{}-{}-{}-{}'.format('AB', lambda_param, alpha_omega_0, alpha_theta_0))
    storefile = os.path.join(storedir, '{}-{}-{}-{}'.format('AB', lambda_param, alpha_omega_0, alpha_theta_0))
    # Log
    log = SummaryWriter(logfile)
    print('Writing logs to {}'.format(logfile))

    theta = np.zeros(value_function.maxSize)
    omega = np.zeros(value_function.maxSize)
    MSE_errors = []
    MSBPE_errors = []
    for i, episode in enumerate(data):
        MSE_error = MSE_function(theta)
        MSBPE_error = MSBPE_function(theta)
        MSE_errors.append(MSE_error)
        MSBPE_errors.append(MSBPE_error)

        log.add_scalar('MSE error', MSE_error, i)
        log.add_scalar('MSBPE error', MSBPE_error, i)

        if (i > nepisodes) or (MSBPE_error > 50):
            break

        # if i % 10 == 0:
        #     print('episode %d, MSE %f, MSBPE %f' % (i, MSE_error, MSBPE_error))
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
            iteration = i * len(episode['rewards']) + idx

            # if i > 1000:
            #     alpha_omega = alpha_omega_0 / np.sqrt(i)
            #     alpha_theta = alpha_theta_0 / np.sqrt(i)
            # else:
            alpha_omega = alpha_omega_0
            alpha_theta = alpha_theta_0

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
            expected_phiprime = 0
            weighted_phiprime = 0
            if newPosition != POSITION_MAX:
                for a in ACTIONS:
                    phiprime = value_function.feature(newPosition, newVelocity, a)
                    expected_phiprime += get_target_prob(newPosition, newVelocity, a) * phiprime
                    weighted_phiprime += min(get_target_prob(newPosition, newVelocity, a),
                                             get_behavior_prob(newPosition, newVelocity, a)) * phiprime
                # expected_phiprime = np.sum(
                #     [get_target_prob(newPosition, newVelocity, a) * value_function.feature(newPosition, newVelocity, a)
                #      for a in ACTIONS], axis=0)
                # weighted_phiprime = np.sum(
                #     [min(get_target_prob(newPosition, newVelocity, a), get_behavior_prob(newPosition, newVelocity, a))
                #      * value_function.feature(newPosition, newVelocity, a)
                #      for a in ACTIONS], axis=0)

            V = np.dot(old_theta, expected_phiprime)
            td_target = reward + DISCOUNT_FACTOR * V
            delta = td_target - q

            omega = old_omega + alpha_omega * (delta * e - np.dot(old_omega, phi) * phi)

            theta = old_theta + alpha_theta * (delta * e - DISCOUNT_FACTOR * np.dot(old_omega, e) *
                                               (expected_phiprime - lambda_param * weighted_phiprime))

    print('saving errors in {}'.format(storefile))
    errors = {
        'MSE errors': MSE_errors,
        'MSBPE errors': MSBPE_errors
    }
    with open(storefile, 'w') as f:
        json.dump(errors, f)
    return MSE_errors, MSBPE_errors


def GQ(value_function, data, lambda_param, obj_function, alpha_omega_0, alpha_theta_0, nepisodes, logdir, storedir):

    # Log
    logfile = os.path.join(logdir, '{}-{}-{}-{}'.format('GQ', lambda_param, alpha_omega_0, alpha_theta_0))
    storefile = os.path.join(storedir, '{}-{}-{}-{}'.format('GQ', lambda_param, alpha_omega_0, alpha_theta_0))
    log = SummaryWriter(logfile)
    print('Writing logs to {}'.format(logfile))

    theta = np.zeros(value_function.maxSize)
    omega = np.zeros(value_function.maxSize)
    MSE_errors = []
    for i, episode in enumerate(data):
        MSE_error = obj_function(theta)
        MSE_errors.append(MSE_error)

        log.add_scalar('MSE error', MSE_error, i)

        if (i > nepisodes) or (MSE_error > 10):
            break

        # if i % 10 == 0:
        #     print('episode %d, error %f' % (i, MSE_error))
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
            iteration = i * len(episode['rewards']) + idx

            # if i > 1000:
            #     alpha_omega = alpha_omega_0 / np.sqrt(i)
            #     alpha_theta = alpha_theta_0 / np.sqrt(i)
            # else:
            alpha_omega = alpha_omega_0
            alpha_theta = alpha_theta_0

            old_theta = theta.copy()
            old_omega = omega.copy()

            phi = value_function.feature(position, velocity, action)
            q = np.dot(old_theta, phi)

            kappa = get_target_prob(position, velocity, action) / get_behavior_prob(position, velocity, action)

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

    print('saving errors in {}'.format(storefile))
    errors = {
        'MSE errors': MSE_errors
    }
    with open(storefile, 'w') as f:
        json.dump(errors, f)
    return MSE_errors


def extragradient_off_policy(value_function, data, lambda_param, return_type, obj_function, alpha_omega_0, alpha_theta_0, nepisodes):
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
            iteration = i * len(episode['rewards']) + idx

            if i % 10 == 0:
                alpha_omega = alpha_omega_0 / np.sqrt(i + 1)
                alpha_theta = alpha_theta_0 / np.sqrt(i + 1)

            # alpha_omega = alpha_omega_0
            # alpha_theta = alpha_theta_0

            old_theta = theta.copy()
            old_omega = omega.copy()

            phi = value_function.feature(position, velocity, action)
            q = np.dot(old_theta, phi)

            if return_type == 'TB':
                kappa = get_target_prob(position, velocity, action)
            elif return_type == 'Retrace':
                kappa = min(
                    [1, get_target_prob(position, velocity, action) / get_behavior_prob(position, velocity, action)])

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

            omega_m = old_omega + alpha_omega * (delta * e - np.dot(old_omega, phi) * phi)

            theta_m = old_theta - alpha_theta * np.dot(old_omega, e) * \
                                (DISCOUNT_FACTOR * expected_phiprime - phi)
            q_m = np.dot(theta_m, phi)
            V_m = np.dot(theta_m, expected_phiprime)
            td_target_m = reward + DISCOUNT_FACTOR * V_m
            delta_m = td_target_m - q_m

            omega = old_omega + alpha_omega * (delta_m * e - np.dot(omega_m, phi) * phi)

            theta = old_theta - alpha_theta * np.dot(omega_m, e) * \
                                (DISCOUNT_FACTOR * expected_phiprime - phi)
        if i > nepisodes:
            break
    return errors




