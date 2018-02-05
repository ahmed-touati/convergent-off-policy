from __future__ import print_function
import numpy as np
from TileCoding import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import os

# all possible actions
ACTION_REVERSE = -1
ACTION_ZERO = 0
ACTION_FORWARD = 1
# order is important
ACTIONS = [ACTION_REVERSE, ACTION_ZERO, ACTION_FORWARD]

# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

# use optimistic initial value, so it's ok to set epsilon to 0
EPSILON = 0
import pdb
# take an @action at @position and @velocity
# @return: new position, new velocity, reward (always -1)

DISCOUNT_FACTOR = 0.999


def takeAction(position, velocity, action):
    newVelocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    newVelocity = min(max(VELOCITY_MIN, newVelocity), VELOCITY_MAX)
    newPosition = position + newVelocity
    newPosition = min(max(POSITION_MIN, newPosition), POSITION_MAX)
    reward = -1.0
    if newPosition == POSITION_MIN:
        newVelocity = 0.0
    return newPosition, newVelocity, reward


# wrapper class for state action value function
class ValueFunction:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @maxSize: the maximum # of indices
    def __init__(self, numOfTilings=10, maxSize=96):
        self.maxSize = maxSize
        self.numOfTilings = numOfTilings

        # divide step size equally to each tiling
        # self.stepSize = stepSize / numOfTilings

        self.hashTable = IHT(maxSize)

        # position and velocity needs scaling to satisfy the tile software
        self.positionScale = self.numOfTilings / (POSITION_MAX - POSITION_MIN)
        self.velocityScale = self.numOfTilings / (VELOCITY_MAX - VELOCITY_MIN)

    # get indices of active tiles for given state and action
    def getActiveTiles(self, position, velocity, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        activeTiles = tiles(self.hashTable, self.numOfTilings,
                            [self.positionScale * position, self.velocityScale * velocity],
                            [action])
        return activeTiles

    # estimate the value of given state and action
    def value(self, theta, position, velocity, action):
        if position == POSITION_MAX:
            return 0.0
        activeTiles = self.getActiveTiles(position, velocity, action)
        return np.sum(theta[activeTiles])

    # learn with given state, action and target
    # def learn(self, position, velocity, action, target):
    #     activeTiles = self.getActiveTiles(position, velocity, action)
    #     estimation = np.sum(self.weights[activeTiles])
    #     delta = self.stepSize * (target - estimation)
    #     for activeTile in activeTiles:
    #         self.weights[activeTile] += delta

    def feature(self, position, velocity, action):
        phi = np.zeros(self.maxSize)
        if position != POSITION_MAX:
            activeTiles = self.getActiveTiles(position, velocity, action)
            phi[activeTiles] = 1
        return phi

    def compute_MSE(self, theta, test_points, true_q):
        error = 0.0
        for position, velocity, action, q in zip(test_points['positions'], test_points['velocities'],
                                                 test_points['actions'], true_q):
            error += (self.value(theta, position, velocity, action) - q)**2
        error = error/np.sum(true_q**2)
        return error



    # get # of steps to reach the goal under current state value function
    def costToGo(self, position, velocity):
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))
        return -np.max(costs)


# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def getAction(position, velocity, valueFunction):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(valueFunction.value(position, velocity, action))
    return np.argmax(values) - 1


def behavior_policy(position, velocity):
    if velocity > 0:
        return [1./300, 1./300, 298./300]
    else:
        return [298./300, 1./300, 1./300]


def get_behavior_prob(position, velocity, action):
    if velocity > 0:
        return [1./300, 1./300, 298./300][action+1]
    else:
        return [298./300, 1./300, 1./300][action+1]


def target_policy(position, velocity):
    if velocity > 0:
        return [0.1, 0.1, 0.8]
    else:
        return [0.8, 0.1, 0.1]


def get_target_prob(position, velocity, action):
    if velocity > 0:
        return [0.1, 0.1, 0.8][action+1]
    else:
        return [0.8, 0.1, 0.1][action+1]


