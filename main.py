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

env = FlappyBird()

state_size = 4
action_size = 2

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma  = 0.99 #future discount
        self.epsilon = 1.0 #epsilon greedy
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = self.build_model()
        self.act_predictions = [0, 0]

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.0015))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            a = random.randint(0,10)
            if a == 0:
                return 1
            else:
                return 0
        act_values = self.model.predict(state, verbose=0) #for every input from action_size
        self.act_predictions = act_values[0]
        return np.argmax(act_values[0])

    def replay(self):
        batch = random.sample(self.memory, 64)
        for state, action, reward, next_state, done in batch:
                q = reward
                if not done:
                        q = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
                q_f = self.model.predict(state, verbose=0)
                q_f[0][action] = q
                self.model.fit(state, q_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)


agent = DQNAgent(state_size, action_size)
model_name = 'nowy-3400'
agent.load(model_name)
agent.epsilon = 0.01
score = 0
record = []
for e in range(0, 101):
    print('episode: ' + str(e) + ', epsilon: ' + str(agent.epsilon))
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    score = 0
    env.done = 0
    for time in range(10000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        action = agent.act(state)
        env.render(agent.act_predictions[0], agent.act_predictions[1])
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -500
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if env.done == 1:
            print('score:' + str(score))
            record.append(score)
            break
    #if len(agent.memory) > 64:
    #     agent.replay()
    print('----------------------------')
