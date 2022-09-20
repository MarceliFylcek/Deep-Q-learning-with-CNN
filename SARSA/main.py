import sys, pygame
import gym
import numpy as np
import random
import os

pygame.init()
env = gym.make('MountainCar-v0')

#states = [x, velocity]

#termination -> (x>=0.5)
#obs high [0.6 0.07]
#obs low [-1.2 -0.07]
#delta [0.01 0.001]
#num of discrete values: [180 140]
#num of actions: 3
#Q table [180, 140, 3] = 75 600 values
#indexing -1.2 -1.19 -1.18 ... 0.6
# +1.2 * 100
# 0 1 2
# -0.070 -0.069 -0.068 ... 0.070
# +0.070 * 1000
# 0 1 2


class SARSA_Agent:
    def __init__(self):
        self.Q_table = np.zeros([180, 140, 3])
        self.learning_rate = 0.05
        self.epsilon = 1
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.02
        self.discount = 1

    def act(self, state):
        if np.random.rand() <=  self.epsilon:
            return random.randint(0, 2)
        i_x, i_v = self.get_index(state)
        return np.argmax(self.Q_table[i_x][i_v])

    def get_index(self, state):
        x = round(state[0], 2)
        v = round(state[1], 3)
        i_x = int((x + 1.2) * 100)
        i_v = int((v + 0.07) * 100)
        return i_x, i_v

    def learn(self, state_A, action_A, reward, state_B, action_B):
        i_xA, i_vA = self.get_index(state_A)
        i_xB, i_vB = self.get_index(state_B)
        calc_Q = reward + self.discount*self.Q_table[i_xB][i_vB][action_B]
        present_Q = self.Q_table[i_xA][i_vA][action_A]
        updated_Q = present_Q + self.learning_rate*(calc_Q-present_Q)
        self.Q_table[i_xA][i_vA][action_A] = updated_Q

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        with open('Q_table.npy', 'wb') as f:
            np.save(f, self.Q_table)

    def load(self):
        with open('Q_table.npy', 'rb') as f:
            self.Q_table = np.load(f)


agent = SARSA_Agent()
agent.load()
agent.epsilon = 0
steps_num = 0
for e in range(0, 1000000):
    state_A = env.reset()
    done = 0
    score = 0

    # if (e+1)%5000 == 0:
    #     agent.save()
    #     print('SAVED')

    for t in range(0, 100000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        # if e>100000:
        #     env.render()
        env.render()

        action_A = agent.act(state_A)
        state_B, reward, done, info = env.step(action_A)
        score += reward
        action_B = agent.act(state_B)
        agent.learn(state_A, action_A, reward, state_B, action_B)
        state_A = state_B
        steps_num += 1
        if done:
            # os.system('cls') #WINDOWS
            # os.system('clear') #LINUX
            print('Episode: '+ str(e))
            print('Epsilon: ' + str(agent.epsilon))
            print('Score:' +str(score))
            print('Number of steps: ' + str(steps_num))
            print('-----------------')
            break
