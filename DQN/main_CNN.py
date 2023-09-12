from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
import os
import random
import numpy as np
from collections import deque
from bird import FlappyBird
import matplotlib.pyplot as plt
#from google.colab import drive
import pickle
import copy

#os.environ['SDL_VIDEODRIVER']='dummy'
#drive.mount('/content/gdrive')
#CZYSZCZENIE KONSOLI
# clear = lambda: os.system('cls')   #WINDOWS
clear = lambda: os.system('clear')   #LINUX

env = FlappyBird(True)
env.return_image = True

action_size = 2
frame_size = [int(576*env.res_ratio*env.image_resize), int(432*env.res_ratio*env.image_resize), 1]
state_size = [3, frame_size[0], frame_size[1], 1]

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = 50000 #10000
        self.memory = deque(maxlen=self.buffer_size)
        self.gamma  = 0.97 #future discount
        self.epsilon = 1.0 #epsilon greedy
        self.epsilon_decay = 0.99992#0.99996
        self.epsilon_min = 0.02
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_copy = 4000
        self.act_predictions = [0, 0]

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8,8), strides=4 ,input_shape=state_size, activation='relu'))
        model.add(Conv2D(64, (4,4), strides=2 ,input_shape=state_size, activation='relu'))
        model.add(Conv2D(64, (3,3), strides=1 ,input_shape=state_size, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, input_dim=state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.0004))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, action_size-1)

        s = np.reshape(state, [1, 3, state_size[1], state_size[2], 1])/255
        act_values = self.model.predict(s, verbose=0) #for every input from action_size
        self.act_predictions = act_values[0]
        print("Akcja 0: " + str(act_values[0][0]))
        print("Akcja 1: " + str(act_values[0][1]))
        return np.argmax(act_values[0])

    def replay(self, trainings):
        batch_size = 32
        batch = random.sample(self.memory, batch_size)

        state_batch = []
        next_state_batch = []
        action_batch, reward_batch, done_batch = [], [], []

        for i in range(batch_size):
            state_batch.append(batch[i][0])
            action_batch.append(batch[i][1])
            reward_batch.append(batch[i][2])
            next_state_batch.append(batch[i][3])
            done_batch.append(batch[i][4])

        state_batch = np.array(state_batch)
        
        state_batch = state_batch/255
        next_state_batch = np.array(next_state_batch)
        next_state_batch = next_state_batch/255
        q = self.model.predict(state_batch)
        q_target = self.target_model.predict(next_state_batch)
    
        for i in range(batch_size):
            q_bellman = reward_batch[i]
            if not done_batch[i]:
                q_bellman = reward_batch[i] + self.gamma * np.amax(q_target[i])
            q[i][action_batch[i]] = q_bellman
        self.model.fit(state_batch, q, epochs=1, batch_size=batch_size, verbose=0)
        if trainings%self.target_copy == 0:
            print("TARGET_COPIED")
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
        self.score_hist    = []        #historia wynikow podczas ewaluacji
        self.avg_score_hist= []        #historia wszystkich srednich wynikow
        self.steps_target  = 2000       #ilosc klatek podczas ewaluacji

        self.steps         = 0         #licznik stepow

        self.override      = True      #nadpisywanie istniejacych plikow

        self.q_set = []
        self.q_set_collected = False   #zebranie setu do ewaluacji sredniego Q
        self.q_set_size      = 5000    #liczba stanow do zapamietania
        self.avg_q_hist      = []
        self.saves = 0
        self.steps_hist = []
        self.avg_steps_hist = []

    def get_q_set(self, buffer):
        ran_mem = random.sample(buffer, self.q_set_size)

        for i in range(self.q_set_size):
            self.q_set.append(ran_mem[i][0])

        self.q_set = np.array(self.q_set)
        self.q_set = self.q_set/255
        self.q_set_collected = True

    def evaluate_avg_q(self):
        q = agent.model.predict(self.q_set) #zwraca q dla kazdej akcji size x a
        res = np.average(np.max(q, axis = 1))    #srednia(max q wzdluz wierszy)
        self.avg_q_hist.append(res)


    def evaluate_score(self):
        sum = 0
        sum_t = 0
       

        for i in range(len(self.score_hist)):
            sum += self.score_hist[i]
          

        self.avg_score_hist.append(sum/len(self.score_hist))
        self.score_hist = []

       

        for i in range(len(train_score_hist)):
            sum_t += train_score_hist[i]

        if len(train_score_hist) != 0:
            train_score_avg_hist.append(sum_t/len(train_score_hist))
        else:
            train_score_avg_hist.append(0)


    def save(self, e):
        path = '/content/gdrive/My Drive/dqnCNN2/flappy_cnn7' + '.txt'
        if not os.path.exists(path) or self.override == True :
            with open(path, 'w') as fp:         #zapisz wyniki do txt
                for i in range(len(self.avg_score_hist)):
                    fp.write("%s, " % self.avg_score_hist[i])
                    fp.write("%s, " % self.avg_q_hist[i])
                    fp.write("%s\n" % train_score_avg_hist[i])
            print('Saved')
            if self.saves%3==0:
              agent.save("/content/gdrive/My Drive/dqnCNN2/fl"+str(trainings)+"e"+str(ev.prev_epsilon))

              data = [agent.memory, self.q_set, agent.epsilon, self.avg_score_hist, self.avg_q_hist, train_score_hist, train_score_avg_hist, trainings, e]

              if self.saves%2==0:
                  d_path = ("/content/gdrive/My Drive/dqnCNN2/data_p.dat")
              else:
                  d_path = ("/content/gdrive/My Drive/dqnCNN2/data.dat")
  
              with open(d_path, "wb") as f:
                  pickle.dump(data, f)
            self.saves +=1

        else:
            print("File already exists")

    def load(self):
        agent.load("/content/gdrive/My Drive/dqnCNN2/fl124001e0.019998930434274764") #<------------
        self.q_set_collected = True
        d_path = ("/content/gdrive/My Drive/dqnCNN2/data_p.dat")
        data = []
        with open(d_path, "rb") as f:
            data = pickle.load(f)
        print("LOADING")
        agent.memory = data[0]
        self.q_set = data[1]
        agent.epsilon = data [2]
        self.avg_score_hist = data[3]
        self.avg_q_hist = data[4]
        global train_score_hist
        train_score_hist = data [5]
        global train_score_avg_hist
        train_score_avg_hist = data [6]
        global trainings
        trainings = data[7]
        global e_start
        e_start = data[8]


def make_move(action):
     done = 0
     rewards = 0
     global recent_frames
     for i in range(3):
        if i > 0:
            action = 0
        frame, reward, done, _ = env.step(action) #get next frame
        frame = np.expand_dims(frame, axis=2)
        recent_frames.append(frame)
        rewards += reward
        if done:
            break
     return copy.deepcopy(recent_frames), rewards, done, _


trainings = 0
epoch_size = 1000
agent = DQNAgent(state_size, action_size)
ev = Evaluator()
train_score_hist = deque(maxlen=25)
train_score_avg_hist = []


e_start = 0
recent_frames = deque(maxlen=3)
ev.load()


for e in range(e_start, 10000000000):
    frame = env.reset()
    frame = np.expand_dims(frame, axis=2) # [r,c] -> [r,c,1]
    recent_frames.append(frame)
    recent_frames.append(frame)
    recent_frames.append(frame)

    state = recent_frames

    score = 0
    env.done = 0
    for time in range(0, 600):
        #clear()
        print("Episode: "+ str(e))
        print("Trainings: "+ str(trainings))
        print("Epsilon " + str(agent.epsilon))
        print("Evaluation: "+ str(ev.mode))
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT: sys.exit()


        action = agent.act(state)
        #env.render(0, 0)

        next_state, reward, done, _ = make_move(action) #get next frame

        reward = reward if not done else -1
        score += reward

        if not Evaluator.mode:
            agent.remember(state, action, reward, next_state, done)

        state = copy.deepcopy(next_state)

        if trainings%epoch_size == 0 and len(agent.memory) == agent.buffer_size:       #EWALUACJA
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
                agent.epsilon = ev.prev_epsilon
                agent.replay(trainings)
                trainings += 1
                ev.save(e)
                break

            else:
                ev.steps += 1

        if len(agent.memory) == agent.buffer_size and not Evaluator.mode:
            agent.replay(trainings)
            print("replay")
            trainings += 1

        if env.done == 1:
            if Evaluator.mode:
                ev.score_hist.append(score)
                print("Appended score")
                print(score)

            else:
                if len(agent.memory) == agent.buffer_size and not Evaluator.mode:
                    print('score:' + str(score))
                    train_score_hist.append(score)
            break #zakoncz episod


        

       
