# -*- coding: utf-8 -*-
"""
Q-Learning Algorithm with cProfile Integration
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import cProfile
import pstats


BOARD_ROWS = 5
BOARD_COLS = 5
START = (0, 0)
WIN_STATE = (4, 4)
HOLE_STATE = [(1,0),(3,1),(4,2),(1,3)]


class State:
    def __init__(self, state=START):
        self.state = state
        self.isEnd = False        

    def getReward(self):
        if self.state in HOLE_STATE:
            return -5
        elif self.state == WIN_STATE:
            return 1       
        else:
            return -1

    def isEndFunc(self):
        self.isEnd = self.state == WIN_STATE or self.state in HOLE_STATE

    def nxtPosition(self, action):     
        if action == 0:                
            nxtState = (self.state[0] - 1, self.state[1])  # up             
        elif action == 1:
            nxtState = (self.state[0] + 1, self.state[1])  # down
        elif action == 2:
            nxtState = (self.state[0], self.state[1] - 1)  # left
        else:
            nxtState = (self.state[0], self.state[1] + 1)  # right

        if 0 <= nxtState[0] <= 4 and 0 <= nxtState[1] <= 4:
            return nxtState 
        return self.state 


class Agent:
    def __init__(self):
        self.states = []
        self.actions = [0,1,2,3]
        self.State = State()
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.isEnd = self.State.isEnd
        self.plot_reward = []
        self.Q = {}
        self.new_Q = {}
        self.rewards = 0
        
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for k in range(len(self.actions)):
                    self.Q[(i, j, k)] = 0
                    self.new_Q[(i, j, k)] = 0

    def Action(self):
        rnd = random.random()
        mx_nxt_reward = -10
        action = None
        i, j = self.State.state

        if rnd > self.epsilon:
            for k in self.actions:
                nxt_reward = self.Q[(i,j,k)]
                if nxt_reward >= mx_nxt_reward:
                    action = k
                    mx_nxt_reward = nxt_reward
        else:
            action = np.random.choice(self.actions)

        position = self.State.nxtPosition(action)
        return position, action

    def Q_Learning(self, episodes):
        x = 0
        while x < episodes:
            if self.isEnd:
                reward = self.State.getReward()
                self.rewards += reward
                self.plot_reward.append(self.rewards)
                i, j = self.State.state
                for a in self.actions:
                    self.new_Q[(i,j,a)] = round(reward, 3)
                self.State = State()
                self.isEnd = self.State.isEnd
                self.rewards = 0
                x += 1
            else:
                mx_nxt_value = -10
                next_state, action = self.Action()
                i, j = self.State.state
                reward = self.State.getReward()
                self.rewards += reward
                for a in self.actions:
                    nxtStateAction = (next_state[0], next_state[1], a)
                    q_value = (1 - self.alpha) * self.Q[(i,j,action)] + \
                              self.alpha * (reward + self.gamma * self.Q[nxtStateAction])
                    if q_value >= mx_nxt_value:
                        mx_nxt_value = q_value

                self.State = State(state=next_state)
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd
                self.new_Q[(i,j,action)] = round(mx_nxt_value, 3)

            self.Q = self.new_Q.copy()

    def plot(self, episodes):
        plt.plot(self.plot_reward)
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward")
        plt.title("Reward over Episodes")
        plt.show()

    def showValues(self):
        for i in range(BOARD_ROWS):
            print('-----------------------------------------------')
            out = '| '
            for j in range(BOARD_COLS):
                mx_nxt_value = max(self.Q[(i,j,a)] for a in self.actions)
                out += str(mx_nxt_value).ljust(6) + ' | '
            print(out)
        print('-----------------------------------------------')


def main():
    ag = Agent()
    episodes = 10000
    ag.Q_Learning(episodes)
    ag.plot(episodes)
    ag.showValues()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(20)
