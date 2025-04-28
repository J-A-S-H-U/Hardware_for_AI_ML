import numpy as np
import matplotlib.pyplot as plt
import time

# Set the rows and columns length
BOARD_ROWS, BOARD_COLS = 5, 5
actions = [0, 1, 2, 3]  # up, down, left, right
move = np.array([[ -1, 0], [1, 0], [0, -1], [0, 1]])

class Agent:
    def __init__(self):
        self.Q = np.zeros((BOARD_ROWS, BOARD_COLS, len(actions)), dtype=np.float32)
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.plot_reward = []

    def choose_action(self, state):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.Q[state[0], state[1], :])
        else:
            return np.random.choice(actions)

    def step(self, state, action):
        next_state = state + move[action]
        next_state = np.clip(next_state, [0, 0], [BOARD_ROWS-1, BOARD_COLS-1])
        return tuple(next_state)

    def get_reward(self, state):
        if state == (4, 4):
            return 1
        if state in [(1, 0), (3, 1), (4, 2), (1, 3)]:
            return -5
        return -1

    def train(self, episodes=10000):
        for episode in range(episodes):
            state = np.array([0, 0])
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state = self.step(state, action)
                reward = self.get_reward(next_state)
                total_reward += reward

                done = reward == 1 or reward == -5

                best_next_q = np.max(self.Q[next_state[0], next_state[1], :])
                td_target = reward + self.gamma * best_next_q
                td_error = td_target - self.Q[state[0], state[1], action]
                self.Q[state[0], state[1], action] += self.alpha * td_error

                state = np.array(next_state)
            self.plot_reward.append(total_reward)

    def plot(self):
        plt.plot(self.plot_reward)
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.title('Training Progress')
        plt.grid()
        plt.show()

    def show_values(self):
        for i in range(BOARD_ROWS):
            print('------------------------------------------------')
            out = '| '
            for j in range(BOARD_COLS):
                max_q = np.max(self.Q[i, j, :])
                out += str(round(max_q, 2)).ljust(6) + ' | '
            print(out)
        print('------------------------------------------------')

if __name__ == "__main__":
    agent = Agent()
    episodes = 10000
    agent.train(episodes)
    agent.plot()
    agent.show_values()
