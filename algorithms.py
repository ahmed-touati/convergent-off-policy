from utils import policy_evaluation
import numpy as np
import pdb

from lib.envs.baird import Baird


def td(env, policy, discount_factor=0.6, lambda_param=0.8, alpha=0.5, num_episodes=100):
    print 'td lambda'
    nS = env.nS
    nA = env.nA
    _, trueV = policy_evaluation(env, policy, discount_factor=discount_factor)

    V = np.zeros(nS)
    errors = []
    for episode in range(num_episodes):
        state = env.reset()
        e = np.zeros(nS)
        while True:
            action_probs = policy[state]
            action = np.random.choice(np.arange(nA), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            td_target = reward + discount_factor * V[next_state]
            delta = td_target - V[state]
            e *= lambda_param * discount_factor
            e[state] += 1

            V += alpha * delta * e
            if done:
                break

            state = next_state
        error = np.sqrt(np.mean(np.power(trueV - V, 2)))
        errors.append(error)
    env.close()
    return errors


def off_policy(env, value_function, target_policy, behaviour_policy, trueQ, operator, theta=None,
               discount_factor=0.6, lambda_param=0.8, alpha=0.5, num_episodes=100, decay=True, type='TB'):
    nS = env.nS
    nA = env.nA

    alpha_0 = alpha
    if theta is None:
        theta = np.zeros(value_function.param_shape)
    count = 0
    errors = []

    for episode in range(num_episodes):

        q = np.zeros([nS, nA])
        for s in range(nS):
            for a in range(nA):
                q[s, a] = np.dot(theta, value_function.feature(s, a))
        if trueQ is not None:
            error = np.sqrt(np.sum(operator.mu.reshape((nS, nA)) * np.power(trueQ - q, 2)))
        else:
            error = operator.compute_RMSPBE(q)
        errors.append(error)

        if episode % 500 == 0:
            print "episode %d, error %f" % (episode, error)

        state = env.reset()

        # initialising eligibility traces
        e = np.zeros(shape=theta.shape)

        while True:

            action_probs = behaviour_policy[state]
            action = np.random.choice(np.arange(nA), p=action_probs)

            next_state, reward, done, _ = env.step(action)

            feature = value_function.feature(state, action)
            q_s_a = np.dot(theta, feature)

            expected_feature_ns = np.sum([target_policy[next_state, a] * value_function.feature(next_state, a)
                                          for a in np.arange(nA)], axis=0)

            V = np.dot(theta, expected_feature_ns)
            td_target = reward + discount_factor * V
            delta = td_target - q_s_a

            pi_s_a = target_policy[state, action]
            mu_s_a = behaviour_policy[state, action]

            if type == 'TB':
                kappa = pi_s_a
            elif type == 'Retrace':
                kappa = min([1, pi_s_a / mu_s_a])
            else:
                raise ValueError()

            e *= discount_factor * lambda_param * kappa
            e += feature

            theta += alpha * delta * e

            count += 1
            if decay:
                alpha = alpha_0 / np.sqrt(count)

            if isinstance(env, Baird):
                if np.random.binomial(1, 0.01) == 1:
                    done = True

            if done:
                break

            state = next_state

    env.close()
    return errors


def gradient_off_policy(env, value_function, target_policy, behaviour_policy, trueQ, operator, theta=None,
                        discount_factor=0.6, lambda_param=0.8, alpha=0.5, num_episodes=100, decay=True, type='TB', threshold=10):
    nS = env.nS
    nA = env.nA

    alpha_0 = alpha
    if theta is None:
        theta = np.zeros(value_function.param_shape)
    omega = np.zeros(theta.shape)
    theta_polyak = alpha * theta
    alpha_sum = alpha
    count = 0
    errors = []

    for episode in range(num_episodes):
        state = env.reset()

        q = np.zeros([nS, nA])
        for s in range(nS):
            for a in range(nA):
                q[s, a] = np.dot(theta_polyak / alpha_sum, value_function.feature(s, a))
        if trueQ is not None:
            error = np.sqrt(np.sum(operator.mu.reshape((nS, nA)) * np.power(trueQ - q, 2)))
        else:
            error = operator.compute_RMSPBE(q)
        errors.append(error)
        # if error > threshold:
        #     return error

        if episode % 500 == 0:
            print "episode %d, error %f" % (episode, error)

        # initialising eligibility traces
        e = np.zeros(shape=theta.shape)

        while True:
            # if state in [0, nS - 1]:
            #     break
            old_omega = omega.copy()
            old_theta = theta.copy()
            action_probs = behaviour_policy[state]
            action = np.random.choice(np.arange(nA), p=action_probs)

            next_state, reward, done, _ = env.step(action)

            feature_s_a = value_function.feature(state, action)
            q_s_a = np.dot(old_theta, feature_s_a)

            if done:
                expected_feature_ns = 0
            else:
                expected_feature_ns = np.sum([target_policy[next_state, a] * value_function.feature(next_state, a)
                                              for a in np.arange(nA)], axis=0)
            V = np.dot(old_theta, expected_feature_ns)
            td_target = reward + discount_factor * V
            delta = td_target - q_s_a

            pi_s_a = target_policy[state, action]
            mu_s_a = behaviour_policy[state, action]

            if type == 'TB':
                kappa = pi_s_a
            elif type == 'Retrace':
                kappa = min([1, pi_s_a / mu_s_a])
            else:
                raise ValueError()
            e *= discount_factor * lambda_param * kappa
            e += feature_s_a

            omega = old_omega + alpha * (delta * e - np.dot(old_omega, feature_s_a) * feature_s_a)

            theta = old_theta - alpha * np.dot(old_omega, e) * \
                    (discount_factor * expected_feature_ns - feature_s_a)

            theta_polyak += alpha * theta
            alpha_sum += alpha
            count += 1
            if decay:
                alpha = alpha_0 / np.sqrt(count)

            if isinstance(env, Baird):
                if np.random.binomial(1, 0.01) == 1:
                    done = True

            if done:
                break

            state = next_state

    env.close()
    return errors


def online_gradient_off_policy(env, value_function, target_policy, behaviour_policy, trueQ, operator, theta=None,
                               discount_factor=0.6, lambda_param=0.6, alpha=0.5, num_episodes=100, decay=True, type='TB', threshold=10):

    nS = env.nS
    nA = env.nA

    alpha_0 = alpha
    if theta is None:
        theta = np.zeros(value_function.param_shape)
    omega = np.zeros(theta.shape)
    theta_polyak = alpha * theta
    alpha_sum = alpha
    count = 0
    errors = []

    for episode in range(num_episodes):
        state = env.reset()

        q = np.zeros([nS, nA])
        for s in range(nS):
            for a in range(nA):
                q[s, a] = np.dot(theta_polyak / alpha_sum, value_function.feature(s, a))
        if trueQ is not None:
            error = np.sqrt(np.sum(operator.mu.reshape((nS, nA)) * np.power(trueQ - q, 2)))
        else:
            error = operator.compute_RMSPBE(q)
        errors.append(error)
        # if error > threshold:
        #     return error

        if episode % 500 == 0:
            print "episode %d, error %f" % (episode, error)

        # initialising eligibility traces
        e = np.zeros(shape=theta.shape)
        # initialising Dutch traces
        dutch = np.zeros(shape=theta.shape)

        while True:
            old_omega = omega.copy()
            old_theta = theta.copy()
            action_probs = behaviour_policy[state]
            action = np.random.choice(np.arange(nA), p=action_probs)

            next_state, reward, done, _ = env.step(action)

            feature_s_a = value_function.feature(state, action)
            q_s_a = np.dot(old_theta, feature_s_a)

            if done:
                expected_feature_ns = 0
            else:
                expected_feature_ns = np.sum([target_policy[next_state, a] * value_function.feature(next_state, a)
                                              for a in np.arange(nA)], axis=0)
            V = np.dot(old_theta, expected_feature_ns)
            td_target = reward + discount_factor * V
            delta = td_target - q_s_a

            pi_s_a = target_policy[state, action]
            mu_s_a = behaviour_policy[state, action]

            if type == 'TB':
                kappa = pi_s_a
            elif type == 'Retrace':
                kappa = min([1, pi_s_a / mu_s_a])
            else:
                raise ValueError()

            e *= discount_factor * lambda_param * kappa
            e += feature_s_a
            dutch = discount_factor * lambda_param * kappa * dutch + \
                    alpha * (1 - discount_factor * lambda_param * kappa * np.dot(dutch, feature_s_a)) * feature_s_a

            omega = old_omega + delta * dutch - alpha * np.dot(old_omega, feature_s_a) * feature_s_a

            theta = old_theta - alpha * np.dot(old_omega, e) * \
                                (discount_factor * expected_feature_ns - feature_s_a)

            theta_polyak += alpha * theta
            alpha_sum += alpha
            count += 1
            if decay:
                alpha = alpha_0 / np.sqrt(count)

            if isinstance(env, Baird):
                if np.random.binomial(1, 0.01) == 1:
                    done = True

            if done:
                break

            state = next_state
    env.close()
    return errors


def extragradient_tree_backup(env, value_function, target_policy, behaviour_policy, tree_backup_operator, theta=None,
                              discount_factor=0.6, lambda_param=0.8, alpha=0.5, num_episodes=100, decay=True):
    # print 'gradient tree backup ...'
    print alpha
    nS = env.nS
    nA = env.nA

    alpha_0 = alpha
    if theta is None:
        theta = np.zeros(value_function.param_shape)
    omega = np.zeros(theta.shape)
    theta_polyak = alpha * theta
    alpha_sum = alpha
    count = 0
    errors = []

    for episode in range(num_episodes):
        state = env.reset()

        q = np.zeros([nS, nA])
        for s in range(nS):
            for a in range(nA):
                q[s, a] = np.dot(theta_polyak / alpha_sum, value_function.feature(s, a))
        error = tree_backup_operator.compute_RMSPBE(q)
        errors.append(error)

        if episode % 500 == 0:
            print "episode %d, error %f" % (episode, error)

        # initialising eligibility traces
        e = np.zeros(shape=theta.shape)

        while True:
            # if state in [0, nS - 1]:
            #     break
            old_omega = omega.copy()
            old_theta = theta.copy()
            action_probs = behaviour_policy[state]
            action = np.random.choice(np.arange(nA), p=action_probs)

            next_state, reward, done, _ = env.step(action)

            feature_s_a = value_function.feature(state, action)
            q_s_a = np.dot(old_theta, feature_s_a)

            expected_feature_ns = np.sum([target_policy[next_state, a] * value_function.feature(next_state, a)
                                          for a in np.arange(nA)], axis=0)
            V = np.dot(old_theta, expected_feature_ns)
            td_target = reward + discount_factor * V
            delta = td_target - q_s_a

            pi_s_a = target_policy[state, action]

            e *= discount_factor * lambda_param * pi_s_a
            e += feature_s_a

            omega_m = old_omega + alpha * (delta * e - np.dot(old_omega, feature_s_a) * feature_s_a)

            theta_m = old_theta - alpha * np.dot(old_omega, e) * \
                    (discount_factor * expected_feature_ns - feature_s_a)

            q_s_a_m = np.dot(theta_m, feature_s_a)
            V_m = np.dot(theta_m, expected_feature_ns)
            td_target_m = reward + discount_factor * V_m
            delta_m = td_target_m - q_s_a_m

            omega = old_omega + alpha * (delta_m * e - np.dot(omega_m, feature_s_a) * feature_s_a)

            theta = old_theta - alpha * np.dot(omega_m, e) * \
                                  (discount_factor * expected_feature_ns - feature_s_a)

            theta_polyak += alpha * theta
            alpha_sum += alpha
            count += 1
            if decay:
                alpha = alpha_0 / np.sqrt(count)

            if isinstance(env, Baird):
                if np.random.binomial(1, 0.01) == 1:
                    done = True

            if done:
                break

            state = next_state

    env.close()
    return errors