import numpy as np
import random
import matplotlib.pyplot as plt
import time
import cProfile
import pstats

BOARD_ROWS = 5
BOARD_COLS = 5
START = (0, 0)
WIN_STATE = (4, 4)
HOLE_STATE = [(1, 0), (3, 1), (4, 2), (1, 3)]

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
        if self.state == WIN_STATE or self.state in HOLE_STATE:
            self.isEnd = True

    def nxtPosition(self, action):
        if action == 0:
            nxtState = (self.state[0] - 1, self.state[1])
        elif action == 1:
            nxtState = (self.state[0] + 1, self.state[1])
        elif action == 2:
            nxtState = (self.state[0], self.state[1] - 1)
        else:
            nxtState = (self.state[0], self.state[1] + 1)

        if 0 <= nxtState[0] < BOARD_ROWS and 0 <= nxtState[1] < BOARD_COLS:
            return nxtState
        return self.state


class Agent:
    def __init__(self):
        self.states = []
        self.actions = [0, 1, 2, 3]
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
                for k in self.actions:
                    self.Q[(i, j, k)] = 0
                    self.new_Q[(i, j, k)] = 0

    def Action(self):
        rnd = random.random()
        mx_nxt_reward = -10
        action = None
        i, j = self.State.state

        if rnd > self.epsilon:
            for k in self.actions:
                nxt_reward = self.Q[(i, j, k)]
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
                    self.new_Q[(i, j, a)] = round(reward, 3)

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
                    q_value = (1 - self.alpha) * self.Q[(i, j, action)] + self.alpha * (reward + self.gamma * self.Q[nxtStateAction])
                    if q_value >= mx_nxt_value:
                        mx_nxt_value = q_value

                self.State = State(state=next_state)
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd
                self.new_Q[(i, j, action)] = round(mx_nxt_value, 3)

            self.Q = self.new_Q.copy()

    def plot(self, episodes):
        plt.plot(self.plot_reward)
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.title('Q-learning Performance Over Time')
        plt.show()

    def showValues(self):
        for i in range(BOARD_ROWS):
            print('-----------------------------------------------')
            out = '| '
            for j in range(BOARD_COLS):
                mx_nxt_value = -10
                for a in self.actions:
                    nxt_value = self.Q[(i, j, a)]
                    if nxt_value >= mx_nxt_value:
                        mx_nxt_value = nxt_value
                out += str(mx_nxt_value).ljust(6) + ' | '
            print(out)
        print('-----------------------------------------------')


if __name__ == "__main__":
    ag = Agent()
    episodes = 10000

    print("Profiling started...")

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()
    ag.Q_Learning(episodes)
    end_time = time.time()

    profiler.disable()
    profiler.dump_stats("frozen_lake_profile.prof")

    print(f"\n⏱️ Total Training Time: {end_time - start_time:.3f} seconds")
    print(f"⏱️ Average Time per Episode: {(end_time - start_time)/episodes:.6f} seconds")

    stats = pstats.Stats("frozen_lake_profile.prof")
    stats.strip_dirs().sort_stats('cumtime').print_stats(15)

    ag.plot(episodes)
    ag.showValues()
