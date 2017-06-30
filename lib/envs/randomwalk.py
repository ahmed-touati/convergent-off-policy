import numpy as np
from gym.envs.toy_text import discrete

LEFT = 0
RIGHT = 1


class RandomWoldEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=7):
        self.size = size

        nS = size
        nA = 2

        P = {}
        chain = np.arange(nS)
        it = np.nditer(chain)

        is_done = lambda s: s == 0 or s == (nS - 1)
        reward = lambda s: 0.0 if s == (nS - 1) or s == 0 else -1.0

        while not it.finished:
            s = it.iterindex
            P[s] = {a: [] for a in range(nA)}

            # We're stuck in a terminal state
            if is_done(s):
                P[s][LEFT] = [(1.0, s, reward(s), True)]
                P[s][RIGHT] = [(1.0, s, reward(s), True)]
                # Not a terminal state
            else:
                ns_left = s - 1
                ns_right = s + 1
                P[s][LEFT] = [(1.0, ns_left, reward(s), is_done(ns_left))]
                P[s][RIGHT] = [(1.0, ns_right, reward(s), is_done(ns_right))]
            it.iternext()

        # Initial state distribution is uniform
        isd = np.zeros(nS)
        isd[nS // 2] = 1

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(RandomWoldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return
        # print("render")
        print("current state: ", self.state)
