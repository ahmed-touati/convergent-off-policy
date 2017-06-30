import numpy as np
import argparse
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.envs.baird import Baird


def make_argument_parser():
    '''Generic experiment parser.
    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.
    Returns:
        argparse.parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount_factor', type=float, default=0.6)
    parser.add_argument('--lambda_param', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--decay', type=bool, default=True)
    parser.add_argument('--regularization_param', type=float, default=0.1)
    return parser


def get_uniform_policy(nS, nA):
    P = np.ones([nS, nA])/nA
    return P


def get_random_policy(nS, nA):
    """ random policy that outputs actions probabilites """
    A = np.random.rand(nS, nA)
    P = A / np.sum(A, axis=1).reshape(-1, 1)
    return P


def estimate_stationary_distribution(env, behaviour_policy):
    cnt = np.zeros((env.nS, env.nA))

    for episode in range(10000):
        state = env.reset()
        while True:
            action_probs = behaviour_policy[state]
            action = np.random.choice(np.arange(env.nA), p=action_probs)
            cnt[state, action] += 1
            next_state, _, done, _ = env.step(action)
            if isinstance(env, Baird):
                if np.random.binomial(1, 0.01) == 1:
                    done = True
            if done:
                break

            state = next_state
    cnt = cnt.astype('float')
    mu = cnt / cnt.sum()
    mu = mu.reshape(env.nS * env.nA)
    return mu


class OffPolicyOperator():
    def __init__(self, env, target_policy, behaviour_policy, mu, discount_factor=0.6, lambda_param=0.8):
        nA = env.nA
        nS = env.nS
        prod = nS * nA
        self.env = env
        self.target_policy = target_policy
        self.behaviour_policy = behaviour_policy
        self.mu = mu
        self.discount_factor = discount_factor
        self.lambda_param = lambda_param
        self.P_target = np.zeros((prod, prod))
        self.neumann = np.zeros((prod, prod))
        self.R = np.zeros(prod)
        self.projection_operator = np.zeros((prod, prod))

    def compute_operators(self, type='TB'):
        """
            Args:
                target_policy: [S, A] shaped matrix representing the target policy.
                behaviour_policy: [S, A] shaped matrix representing the behaviour policy.
                env: OpenAI env. env.P represents the transition probabilities of the environment.
                    env.P[s][a] is a (prob, next_state, reward, done) tuple.
                discount_factor: gamma discount factor.
                lambda_param: lambda bootstrapping parameter

            Returns:
                [S*A, S*A] shaped matrix Tree backup operator
            """
        nA = self.env.nA
        nS = self.env.nS
        prod = nS * nA

        P_target = np.zeros((nS, nA, nS, nA))
        P_kappa = np.zeros((nS, nA, nS, nA))
        R = np.zeros((nS, nA))

        for s in range(nS):
            for a in range(nA):
                r = 0
                for prob, next_state, reward, done in self.env.P[s][a]:
                    r += prob * reward
                    P_target[s, a, next_state, :] = prob * self.target_policy[next_state]
                    if type == 'TB':
                        P_kappa[s, a, next_state, :] = prob * self.target_policy[next_state] * self.behaviour_policy[next_state]
                    elif type == 'Retrace':
                        P_kappa[s, a, next_state, :] = prob * np.min([self.target_policy[next_state],
                                                                      self.behaviour_policy[next_state]], axis=0)
                    else:
                        raise ValueError()
                R[s, a] = r

        identity = np.eye(prod)
        neumann = np.linalg.pinv(identity - self.lambda_param * self.discount_factor * P_kappa.reshape(prod, prod))
        self.P_target = P_target.reshape(prod, prod)
        self.neumann = neumann
        self.R = R.reshape(prod)

    def compute_projection_operator(self, PHI):
        nA = self.env.nA
        nS = self.env.nS
        prod = nS * nA
        PHI = PHI.reshape((prod, -1))
        D = np.diag(self.mu)
        M = np.dot(PHI.T, np.dot(D, PHI))
        projection = np.dot(PHI, np.dot(np.linalg.pinv(M), np.dot(PHI.T, D)))
        self.projection_operator = projection

    def apply_operator(self, Q):
        nA = self.env.nA
        nS = self.env.nS
        prod = nS * nA
        Q = Q.reshape(prod)
        bellman_Q = self.R + self.discount_factor * np.dot(self.P_target, Q)
        Q_output = Q + np.dot(self.neumann, bellman_Q - Q)
        return Q_output

    def compute_RMSPBE(self, Q):
        projected_Q = np.dot(self.projection_operator, self.apply_operator(Q))
        nA = self.env.nA
        nS = self.env.nS
        prod = nS * nA
        Q = Q.reshape(prod)
        rmspbe = np.sqrt(np.mean(self.mu * (Q - projected_Q)**2))
        return rmspbe


def policy_evaluation(env, policy, discount_factor=0.6, threshold=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        threshold: We stop evaluation once our value function change is less than threshold for all states.
        discount_factor: gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    nA = env.nA
    nS = env.nS
    # Start with a random (all 0) value function
    V = np.zeros(nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            if isinstance(env, CliffWalkingEnv):
                position = np.unravel_index(s, env.shape)
                if env._cliff[tuple(position)]:
                    break
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < threshold:
            break

    # compute state-action value
    Q = np.zeros([nS, nA])
    for s in range(nS):
        for a in range(nA):
            q = 0
            for prob, next_state, reward, done in env.P[s][a]:
                q += prob * (reward + discount_factor * V[next_state])
            Q[s, a] = q
    return Q, V


