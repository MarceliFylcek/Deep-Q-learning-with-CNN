from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os, sys
import random
import numpy as np
import gym
from collections import deque
import pygame
from bird import FlappyBird
import math

env = FlappyBird()

state_size = 2
action_size = 2

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.memory_priorities = deque(maxlen=5000)
        self.gamma  = 0.97 #future discount
        self.epsilon = 1.0 #epsilon greedy
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.02
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.act_predictions = [0, 0]

    def build_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #save memory (action, reward from state A to state B)
        # self.memory_priorities.append(max(self.memory_priorities, default=1)) #set maximum priority for new memories

    # def get_probabilities(self, probabilities):
    #     scaled_prorities = np.array(self.memory_priorities) ** priority_scale #adjust by scaling factor, 1 for full priority, 0 for random
    #     sample_probabilities = scaled_protities / sum(scaled_protities) #probability of single memory depends on magnitude of other priorites
    #     return importance_normalized
    #
    # def get_importance(self, probabilities): #for network training, not to overfit
    #     importance = 1/len(self.memory) * 1/probabilities # b scaling??
    #     importance_normalized = importance / max(importance)  #relative to max importance value
    #     return importance_normalized
    #
    # def sample(self, batch_size, priority_scale):
    #     sample_size = min(len(self.memory), batch_size)    #batch_size for training
    #     sample_probs = get_probabilities(priority_scale)   #gets probabilities from priorities
    #     sample_indicies = random.choices(range(len(self.memory)), k = sample_size, weights = sample_probs) #choses random memories indicies based of probabilities calculated earlier
    #     samples = np.array(self.memory)[sample_indicies] #assignes chosen indicies to correspondent memories
    #     importance = get_importance(sample_probs[sample_indicies]) #calculated weight for training NN
    #     return map(list, zip(*samples)), importance, sample_indicies #returns something ???
    #
    # def set_priorites(self, indices, errors, offset=0.1):  #assigns priorites to memories that went through training
    #     for i,e in zip(indices,erros):
    #         self..priorites[i] = abs(e) + offset #priority depens on error during training

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print('EPSILON')
            return random.randint(0,2)%2
        act_values = self.model.predict(state, verbose=0) #for every input from action_size
        self.act_predictions = act_values[0]
        # actions = [0,0]
        # #more sure on not jumping
        # if self.act_predictions[0] > self.act_predictions[1]:
        #     actions[0] = self.act_predictions[0] * 15
        #     actions[1] = self.act_predictions[1]
        #     actions[0] = (actions[0]/(actions[0]+actions[1]))
        #     actions[1] = 1-actions[0]
        # #more sure on jumping
        # else:
        #     actions[1] = self.act_predictions[1] * 3
        #     actions[0] = self.act_predictions[0]
        #     actions[1] = (actions[1]/(actions[1]+actions[0]))
        #     actions[0] = 1-actions[1]
        # print(actions[0])
        # if np.random.rand() <= actions[0]:
        #    print('down')
        #    action = 0
        # else:
        #    action = 1
        #    print('up')
        return np.argmax(act_values[0])

    def replay(self, e):
        # priority_scale = 1
        # scaled_protities = np.array(self.memory_priorities) ** priotity_scale
        batch_size = 32
        batch = random.sample(self.memory, batch_size)

        state_table = np.zeros((batch_size, state_size))
        next_state_table = np.zeros((batch_size, state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            state_table[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_state_table[i] = batch[i][3]
            done.append(batch[i][4])

        q = self.model.predict(state_table)
        q_target = self.target_model.predict(next_state_table)

        for i in range(batch_size):
            q_bellman = reward[i]
            if not done:
                q_bellman = reward + self.gamma * np.amax(q_target[i])

            q[i][action[i]] = q_bellman #substitude q value for action taken by the value from bellman equation

        self.model.fit(state_table, q, epochs=1, batch_size=batch_size, verbose=0)

        if e%50 == 0:
            self.target_model.set_weights(self.model.get_weights())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)


agent = DQNAgent(state_size, action_size)
agent.load('trained_model')
agent.epsilon = 0
score = 0
record = []
for e in range(1, 10000):
    print('episode: ' + str(e) + ' Epsilon: ' + str(agent.epsilon))
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    score = 0
    env.done = 0
    for time in range(0, 1000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        action = agent.act(state)
        env.render(agent.act_predictions[0], agent.act_predictions[1])
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -1
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if env.done == 1:
            print('score:' + str(score))
            record.append(score)
            break

        # if len(agent.memory) > 5000:
        #     agent.replay(e)

    # if e%100 == 0:
    #     agent.save('trained_model-'+str(e)) #zapisz model
    #     with open(r'trained_model'+str(e)+'.txt', 'w') as fp: #zapisz wyniki do txt
    #         for item in record:
    #             # write each item on a new line
    #             fp.write("%s\n" % item)
    #         print('Done')
    # print('----------------------------')
