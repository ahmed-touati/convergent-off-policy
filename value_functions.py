import numpy as np
from sklearn.preprocessing import LabelBinarizer


class TabularValueFunction():
    def __init__(self, nS, nA):
        labels = ['+'.join([str(s), str(a)]) for s in np.arange(nS) for a in np.arange(nA)]
        self.lb = LabelBinarizer()
        self.lb.fit(labels)
        self.nA = nA
        self.nS = nS
        self.param_shape = (nS * nA, )

    def feature(self, s, a):
        return self.lb.transform(['+'.join([str(s), str(a)])]).reshape(-1, )


class BairdValueFunction():
    def __init__(self):
        self.param_shape = (2*8, )
        phis = np.array([
            [2, 0, 0, 0, 0, 0, 0, 1],
            [0, 2, 0, 0, 0, 0, 0, 1],
            [0, 0, 2, 0, 0, 0, 0, 1],
            [0, 0, 0, 2, 0, 0, 0, 1],
            [0, 0, 0, 0, 2, 0, 0, 1],
            [0, 0, 0, 0, 0, 2, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 2]
        ])

        PHI = np.zeros((7, 2, self.param_shape[0]))
        for s in range(7):
            PHI[s, 0, :] = np.concatenate([phis[s], np.zeros(phis.shape[1])])
            PHI[s, 1, :] = np.concatenate([np.zeros(phis.shape[1]), phis[s]])
        self.PHI = PHI

    def feature(self, s, a):
        return self.PHI[s, a, :]


class CounterExampleValueFunction():
    def __init__(self):
        self.param_shape = (2, )
        self.PHI = np.zeros((2, 2, 2))
        self.PHI[0, 0, :] = np.array([1, 0])
        self.PHI[0, 1, :] = np.array([0, 1])
        self.PHI[1, 0, :] = np.array([2, 0])
        self.PHI[1, 1, :] = np.array([0, 2])

    def feature(self, s, a):
        return self.PHI[s, a, :]


class CoordinateValuefunction():
    def __init__(self, nS, nA, shape):
        self.nA = nA
        self.nS = nS
        self.param_shape = (8,)
        self.grid_shape = shape
        PHI = np.zeros((nS, nA, 8))
        for s in range(nS):
            x, y = np.unravel_index(s, shape)
            PHI[s, 0, :] = np.array([x, y, 0, 0, 0, 0, 0, 0])
            PHI[s, 1, :] = np.array([0, 0, x, y, 0, 0, 0, 0])
            PHI[s, 2, :] = np.array([0, 0, 0, 0, x, y, 0, 0])
            PHI[s, 3, :] = np.array([0, 0, 0, 0, 0, 0, x, y])
        self.PHI = PHI

    def feature(self, s, a):
        return self.PHI[s, a, :]


class RandWalkValueFunction():
    def __init__(self, nS, nA, type=None):
        self.nA = nA
        self.nS = nS
        if type == 'tabular':
            self.param_shape = (2 * 5, )
            phis = np.array([[1, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0],
                                  [0, 0, 1, 0, 0],
                                  [0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1]])

        elif type == 'inverted':
            self.param_shape = (2 * 5, )
            phis = 0.5 * np.ones((5, 5)) - 0.5 * np.identity(5)
        elif type == 'dependant':
            self.param_shape = (2 * 3, )
            phis = np.array([[1, 0, 0],
                                  [1./np.sqrt(2), 1./np.sqrt(2), 0],
                                  [1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)],
                                  [0, 1./np.sqrt(2), 1./np.sqrt(2)],
                                  [0, 0, 1]])
        else:
            raise ValueError('type should take value in {tabular, inverted, dependant}')

        PHI = np.zeros((nS, nA, self.param_shape[0]))
        for s in range(1, nS - 1):
            PHI[s, 0, :] = np.concatenate([phis[s - 1], np.zeros(phis.shape[1])])
            PHI[s, 1, :] = np.concatenate([np.zeros(phis.shape[1]), phis[s - 1]])
        self.PHI = PHI

    def feature(self, s, a):
        return self.PHI[s, a, :]
