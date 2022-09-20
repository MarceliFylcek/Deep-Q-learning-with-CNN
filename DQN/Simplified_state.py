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
import matplotlib.pyplot as plt

env = FlappyBird()

state_size = 2
action_size = 2

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = 7000
        self.memory = deque(maxlen=self.buffer_size)
        self.memory_priorities = deque(maxlen=5000)
        self.gamma  = 0.97 #future discount
        self.epsilon = 1.0 #epsilon greedy
        self.epsilon_decay = 0.9997
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

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0,2)%2
        act_values = self.model.predict(state, verbose=0) #for every input from action_size
        self.act_predictions = act_values[0]
        return np.argmax(act_values[0])

    def replay(self, e):
        batch_size = 32
        batch = random.sample(self.memory, batch_size)

        state_batch = np.zeros((batch_size, state_size))
        next_state_batch = np.zeros((batch_size, state_size))
        action_batch, reward_batch, done_batch = [], [], []

        for i in range(batch_size):
            state_batch[i] = batch[i][0]
            action_batch.append(batch[i][1])
            reward_batch.append(batch[i][2])
            next_state_batch[i] = batch[i][3]
            done_batch.append(batch[i][4])

        q = self.model.predict(state_batch) #Q value for both actions for all states from the batch
        q_target = self.target_model.predict(next_state_batch) #Q values for both actions for all future states from the batch

        for i in range(batch_size):
            q_bellman = reward_batch[i]
            if not done_batch[i]:
                q_bellman = reward_batch[i] + self.gamma * np.amax(q_target[i])

            q[i][action_batch[i]] = q_bellman #substitude q value for action taken by the value from bellman equation

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
agent.load('models\model-2064')
agent.epsilon = 0.02
steps_count = 0
record = []
epsilon_hist = []
training_epochs = 0
epochs_hist = []
for e in range(2065, 10000):

    print('episode: ' + str(e) + ' Epsilon: ' + str(agent.epsilon))
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    steps_count = 0
    env.done = 0

    for time in range(0, 1000):

        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        env.render(agent.act_predictions[0], agent.act_predictions[1])
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -1

        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        steps_count += 1

        if env.done == 1:
            if len(agent.memory) == agent.buffer_size:
                print('score:' + str(steps_count))
                record.append(steps_count)
                epsilon_hist.append(agent.epsilon)
                epochs_hist.append(training_epochs)
            break


        if len(agent.memory) == agent.buffer_size:
            agent.replay(e)
            training_epochs += 1

        if training_epochs%1000 == 0 and len(agent.memory) == agent.buffer_size:
            agent.save('models\model-'+str(len(record))) #zapisz model
            with open(r'models\'+str(len(record))'+'.txt', 'w') as fp: #zapisz wyniki do txt
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
