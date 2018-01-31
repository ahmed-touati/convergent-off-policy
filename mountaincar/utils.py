from tqdm import tqdm
import numpy as np
from mountain_car import ACTIONS, behavior_policy, takeAction, POSITION_MAX, target_policy, DISCOUNT_FACTOR
import os
import json


def collect_dataset(logdir, ntrials=5, maxsteps=100000):
    all_positions = []
    all_velocities = []
    all_actions = []
    for k in tqdm(range(ntrials)):
        dataset = []
        nsteps = 0
        while nsteps < maxsteps:
            positions = []
            velocities = []
            actions = []
            rewards = []
            # start at a random position around the bottom of the valley
            currentPosition = np.random.uniform(-0.6, -0.4)
            # initial velocity is 0
            currentVelocity = 0.0

            while True:
                nsteps += 1
                currentAction = np.random.choice(ACTIONS, p=behavior_policy(currentPosition, currentVelocity))
                newPosition, newVelocity, reward = takeAction(currentPosition, currentVelocity, currentAction)
                positions.append(currentPosition)
                velocities.append(currentVelocity)
                actions.append(currentAction)
                rewards.append(reward)
                # track new state and action
                all_positions.append(currentPosition)
                all_velocities.append(currentVelocity)
                all_actions.append(currentAction)

                if (newPosition == POSITION_MAX) or (nsteps >= maxsteps):
                    break

                currentPosition = newPosition
                currentVelocity = newVelocity

            episode = {
                'positions': positions,
                'velocities': velocities,
                'actions': actions,
                'rewards': rewards,
            }
            dataset.append(episode)
        np.save(os.path.join(logdir, 'dataset_%i.npy' % k), dataset)
    print(len(all_actions))
    l = len(all_actions)
    all_positions = all_positions[l//2:]
    all_velocities = all_velocities[l//2:]
    all_actions = all_actions[l//2:]
    indices = np.random.randint(0, len(all_actions), size=100)
    test_positions = [all_positions[ind] for ind in indices]
    test_velocities = [all_velocities[ind] for ind in indices]
    test_actions = [all_actions[ind] for ind in indices]

    test_points = {
        'positions': test_positions,
        'velocities': test_velocities,
        'actions': test_actions
    }
    with open(os.path.join(logdir, 'test_points.json'), 'w') as f:
        json.dump(test_points, f)


def estimate_true_q(test_points, logdir, nepisodes=1000):
    true_q = np.zeros(len(test_points['positions']))
    for idx, (position, velocity, action) in tqdm(enumerate(
            zip(test_points['positions'], test_points['velocities'], test_points['actions']))):
        currentPosition = position
        currentVelocity = velocity
        currentAction = action
        cumreward = 0.0
        for _ in range(nepisodes):
            nsteps = 0
            while True:
                newPosition, newVelocity, reward = takeAction(currentPosition, currentVelocity, currentAction)
                cumreward += np.power(DISCOUNT_FACTOR, nsteps) * reward
                nsteps += 1
                if newPosition == POSITION_MAX:
                    break
                currentPosition = newPosition
                currentVelocity = newVelocity
                currentAction = np.random.choice(ACTIONS, p=target_policy(currentPosition, currentVelocity))
        true_q[idx] = cumreward / nepisodes
    np.save(os.path.join(logdir, 'true_q.npy'), true_q)
