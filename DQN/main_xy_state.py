from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os, sys
import random
import numpy as np
import gym
from collections import deque
import pygame
import math
import matplotlib.pyplot as plt
import os
from bird import FlappyBird

# CZYSZCZENIE KONSOLI
# clear = lambda: os.system('cls')   #WINDOWS
clear = lambda: os.system("clear")  # LINUX

env = FlappyBird(False)

state_size = 2  # <--------
action_size = 2  # <--------


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = 10000  # <--------
        self.memory = deque(maxlen=self.buffer_size)
        self.gamma = (
            0.97  # future discount                                 #<--------
        )
        self.epsilon = (
            1.0  # epsilon greedy                                  #<--------
        )
        self.epsilon_decay = 0.99972  # <--------
        self.epsilon_min = 0.015  # <--------
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_copy = (
            1000  # (co ile stepow taget sie zmienia)         #<-------
        )
        self.act_predictions = [0, 0]

    def build_model(self):
        model = Sequential()
        model.add(
            Dense(48, input_dim=self.state_size, activation="relu")
        )  # <--------
        model.add(Dense(48, activation="relu"))  # <--------
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))  # <--------
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # save memory (action, reward from state A to state B)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, action_size - 1)
        act_values = self.model.predict(
            state, verbose=0
        )  # for every input from action_size
        self.act_predictions = act_values[0]
        return np.argmax(act_values[0])

    def replay(self, trainings):
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

        q = self.model.predict(
            state_batch
        )  # Q value for both actions for all states from the batch
        q_target = self.target_model.predict(
            next_state_batch
        )  # Q values for both actions for all future states from the batch

        for i in range(batch_size):
            q_bellman = reward_batch[i]
            if not done_batch[i]:
                q_bellman = reward_batch[i] + self.gamma * np.amax(q_target[i])

            q[i][
                action_batch[i]
            ] = q_bellman  # substitude q value for action taken by the value from bellman equation

        self.model.fit(
            state_batch, q, epochs=1, batch_size=batch_size, verbose=0
        )

        if trainings % self.target_copy == 0:
            self.target_model.set_weights(self.model.get_weights())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)


class Evaluator:
    mode = False

    def __init__(self):
        self.score_hist = []  # historia wynikow podczas ewaluacji
        self.avg_score_hist = []  # historia wszystkich srednich wynikow
        self.steps_target = 1500  # ilosc klatek podczas ewaluacji
        self.steps = 0  # licznik stepow
        self.override = True  # nadpisywanie istniejacych plikow
        self.q_set_collected = False  # zebranie setu do ewaluacji sredniego Q
        self.q_set_size = 600  # liczba stanow do zapamietania
        self.avg_q_hist = []
        self.steps_hist = []
        self.avg_steps_hist = []

    def get_q_set(self, buffer):
        ran_mem = random.sample(buffer, self.q_set_size)
        q_set = np.zeros((self.q_set_size, state_size))

        for i in range(self.q_set_size):
            q_set[i] = ran_mem[i][0]

        self.q_set = q_set
        self.q_set_collected = True

    def evaluate_avg_q(self):
        q = agent.model.predict(
            self.q_set
        )  # zwraca q dla kazdej akcji size x a
        res = np.average(np.max(q, axis=1))  # srednia(max q wzdluz wierszy)
        self.avg_q_hist.append(res)

    def evaluate_score(self):
        sum = 0
        sum_t = 0
        sum_s = 0
        sum_s_t = 0

        for i in range(len(self.score_hist)):
            sum += self.score_hist[i]
            sum_s += self.steps_hist[i]

        self.avg_score_hist.append(sum / len(self.score_hist))
        self.score_hist = []

        self.avg_steps_hist.append(sum_s / len(self.steps_hist))
        self.steps_hist = []

        for i in range(len(train_score_hist)):
            sum_t += train_score_hist[i]
            sum_s_t += train_steps_hist[i]

        if len(train_score_hist) != 0:
            train_score_avg_hist.append(sum_t / len(train_score_hist))
            train_steps_avg_hist.append(sum_s_t / len(train_score_hist))
        else:
            train_score_avg_hist.append(0)
            train_steps_avg_hist.append(0)

    def save(self):
        path = "DATA_SIMPLE6/results" + ".txt"
        if not os.path.exists(path) or self.override == True:
            with open(path, "w") as fp:  # zapisz wyniki do txt
                for i in range(len(self.avg_score_hist)):
                    fp.write("%s, " % self.avg_score_hist[i])
                    fp.write("%s, " % self.avg_q_hist[i])
                    fp.write("%s, " % train_score_avg_hist[i])
                    fp.write("%s, " % self.avg_steps_hist[i])
                    fp.write("%s\n" % train_steps_avg_hist[i])
            print("Saved")
        else:
            print("File already exists")
        agent.save("DATA_SIMPLE6/model" + str(trainings))


trainings = 0
epoch_size = 500

agent = DQNAgent(state_size, action_size)
# agent.load('DATA_SIMPLE5/model107250')             #load model

ev = Evaluator()
train_score_hist = deque(maxlen=25)  # <------
train_score_avg_hist = []

train_steps_hist = deque(maxlen=25)  # <------
train_steps_avg_hist = []


for e in range(0, 10000):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    env.done = 0
    score = 0
    steps = 0

    for time in range(0, 2000):
        clear()
        print("Episode: " + str(e))
        print("Trainings: " + str(trainings))
        print("Epsilon " + str(agent.epsilon))
        print("Evaluation: " + str(ev.mode))

        env.render(agent.act_predictions[0], agent.act_predictions[1])
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -1  # <---------

        steps += 1
        score += reward
        next_state = np.reshape(next_state, [1, state_size])

        if not Evaluator.mode:
            agent.remember(state, action, reward, next_state, done)

        state = next_state

        if (
            trainings % epoch_size == 0
            and len(agent.memory) == agent.buffer_size
        ):  # EWALUACJA
            if not Evaluator.mode:
                if not ev.q_set_collected:
                    ev.get_q_set(agent.memory)
                ev.evaluate_avg_q()
                Evaluator.mode = True
                print("EWALUACJA")
                ev.prev_epsilon = agent.epsilon
                agent.epsilon = agent.epsilon_min
                break
            elif ev.steps == ev.steps_target:
                if len(ev.score_hist) == 0:
                    ev.score_hist.append(score)
                Evaluator.mode = False
                ev.steps = 0
                ev.evaluate_score()
                print("Koniec ewaluacji")
                ev.save()
                agent.epsilon = ev.prev_epsilon
                agent.replay(trainings)
                trainings += 1
                break

            else:
                ev.steps += 1

        if len(agent.memory) == agent.buffer_size and not Evaluator.mode:
            agent.replay(trainings)
            print("replay")
            trainings += 1

        if done:  # po każdym episodzie
            if Evaluator.mode:
                ev.score_hist.append(score)
                ev.steps_hist.append(steps)
                print("Appended score")
                print(score)

            else:
                if (
                    len(agent.memory) == agent.buffer_size
                    and not Evaluator.mode
                ):
                    print("score:" + str(score))
                    train_score_hist.append(score)
                    train_steps_hist.append(steps)
            break  # zakoncz episod
