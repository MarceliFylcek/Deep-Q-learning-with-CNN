from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import os, sys
import random
import numpy as np
import gym
from collections import deque
import pygame
from bird import FlappyBird
import math
import cv2
import matplotlib.pyplot as plt

env = FlappyBird()
env.return_image = True

action_size = 2
state_size = [int(576*env.image_resize), int(432*env.image_resize), 1]

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.buffer_size = 10000 #
        self.memory = deque(maxlen=self.buffer_size)
        self.memory_priorities = deque(maxlen=5000)
        self.gamma  = 0.97 #future discount
        self.epsilon = 1.0 #epsilon greedy
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.02
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.act_predictions = [0, 0]

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8,8), strides=4 ,input_shape=state_size, activation='relu'))
        model.add(Conv2D(64, (4,4), strides=2 ,input_shape=state_size, activation='relu'))
        model.add(Conv2D(64, (3,3), strides=1 ,input_shape=state_size, activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, input_dim=state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.0008))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0,2)%2

        s = np.reshape(state, [1, state_size[0], state_size[1]])
        act_values = self.model.predict(s, verbose=0) #for every input from action_size
        self.act_predictions = act_values[0]
        return np.argmax(act_values[0])

    def replay(self, e):
        batch_size = 32
        batch = random.sample(self.memory, batch_size)

        state_batch = np.zeros([batch_size, state_size[0], state_size[1], 1])
        next_state_batch = np.zeros([batch_size, state_size[0], state_size[1], 1])
        action_batch, reward_batch, done_batch = [], [], []

        for i in range(batch_size):
            state_batch[i] = batch[i][0]
            action_batch.append(batch[i][1])
            reward_batch.append(batch[i][2])
            next_state_batch[i] = batch[i][3]
            done_batch.append(batch[i][4])

        q = self.model.predict(state_batch)
        q_target = self.target_model.predict(next_state_batch)

        for i in range(batch_size):
            q_bellman = reward_batch[i]
            if not done_batch[i]:
                q_bellman = reward_batch[i] + self.gamma * np.amax(q_target[i])

            q[i][action_batch[i]] = q_bellman

        self.model.fit(state_batch, q, epochs=1, batch_size=batch_size, verbose=0)

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
steps_count = 0
record = []
epsilon_hist = []
epochs = 0
epochs_hist = []
for e in range(1, 10000):
    print('episode: ' + str(e) + ' Epsilon: ' + str(agent.epsilon))
    state = env.reset()
    state = np.expand_dims(state, axis=2) # [r,c] -> [r,c,1]
    steps_count = 0
    env.done = 0
    score = 0
    for time in range(0, 1000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        action = agent.act(state)
        env.render(0, 0)
        next_state, reward, done, _ = env.step(action)
        next_state = np.expand_dims(next_state, axis=2)
        reward = reward if not done else -1
        score += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        steps_count += 1

        if env.done == 1:
            if len(agent.memory) == agent.buffer_size:
                print('Steps:' + str(steps_count))
                print('Score:' + str(score))
                record.append(steps_count)
                epsilon_hist.append(agent.epsilon)
                epochs_hist.append(epochs)
            break

        #cv2.imshow('klatka', state)
        if len(agent.memory) == agent.buffer_size: #5000
            agent.replay(e)
            epochs += 1

        if epochs%500 == 0 and len(agent.memory) == agent.buffer_size:
            agent.save('modele\CNNb-'+str(len(record))) #zapisz model
            with open(r'modele\CNNb'+str(len(record))+'.txt', 'w') as fp: #zapisz wyniki do txt
                for item in record:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            print('Saved')

    print('----------------------------')

    plt.figure(1)
    plt.subplot(311)
    plt.plot(epsilon_hist)
    plt.title('Epsilon')
    plt.subplot(312)
    plt.plot(record)
    plt.title('Steps')
    plt.subplot(313)
    plt.plot(epochs_hist)
    plt.title('Training epochs')
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.8)
    plt.show(block=False)
    plt.pause(.000000001)
