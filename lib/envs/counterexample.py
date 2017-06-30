import numpy as np
from gym.envs.toy_text import discrete

LEFT = 0
RIGHT = 1


class CounterExample(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        nS = 2
        nA = 2

        P = {}

        P[0] = {a: [] for a in range(nA)}
        P[0][LEFT] = [(1.0, 0, 0.0, False)]
        P[0][RIGHT] = [(1.0, 1, 0.0, False)]

        P[1] = {a: [] for a in range(nA)}
        P[1][LEFT] = [(1.0, 0, 0.0, False)]
        P[1][RIGHT] = [(1.0, 1, 0.0, True)]


        # Initial state distribution is uniform
        isd = np.array([0.5, 0.5])

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(CounterExample, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return
        # print("render")
        print("current state: ", self.state)