import numpy as np
import pyopencl as cl
import time
import matplotlib.pyplot as plt

# Common setup
BOARD_ROWS, BOARD_COLS = 5, 5
actions = [0, 1, 2, 3]
move = np.array([[ -1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)
episodes = 10000

# OpenCL setup
def setup_opencl():
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    return context, queue

context, queue = setup_opencl()
mf = cl.mem_flags

# OpenCL kernel
kernel_code = """
__kernel void update_q(
    __global float* Q,
    __global int* moves,
    const int board_rows,
    const int board_cols,
    const float alpha,
    const float gamma,
    const int action,
    const int i,
    const int j,
    const int reward
) {
    int next_i = i + moves[2*action];
    int next_j = j + moves[2*action + 1];

    if (next_i < 0) next_i = 0;
    if (next_i >= board_rows) next_i = board_rows - 1;
    if (next_j < 0) next_j = 0;
    if (next_j >= board_cols) next_j = board_cols - 1;

    int idx = i * board_cols * 4 + j * 4 + action;
    float max_q = -99999.0f;

    for (int a = 0; a < 4; a++) {
        int next_idx = next_i * board_cols * 4 + next_j * 4 + a;
        if (Q[next_idx] > max_q) {
            max_q = Q[next_idx];
        }
    }

    Q[idx] = (1.0f - alpha) * Q[idx] + alpha * (reward + gamma * max_q);
}
"""
program = cl.Program(context, kernel_code).build()

# CPU Agent
class AgentCPU:
    def __init__(self):
        self.Q = np.zeros((BOARD_ROWS, BOARD_COLS, 4), dtype=np.float32)
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1

    def choose_action(self, state):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.Q[state[0], state[1], :])
        else:
            return np.random.choice(actions)

    def get_reward(self, state):
        if tuple(state) == (4, 4):
            return 1
        if tuple(state) in [(1, 0), (3, 1), (4, 2), (1, 3)]:
            return -5
        return -1

    def train(self, episodes=10000):
        rewards = []
        for _ in range(episodes):
            state = np.array([0, 0], dtype=np.int32)
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state = state + move[action]
                next_state = np.clip(next_state, [0, 0], [BOARD_ROWS-1, BOARD_COLS-1])
                reward = self.get_reward(next_state)
                total_reward += reward

                best_next_q = np.max(self.Q[next_state[0], next_state[1], :])
                td_target = reward + self.gamma * best_next_q
                td_error = td_target - self.Q[state[0], state[1], action]
                self.Q[state[0], state[1], action] += self.alpha * td_error

                if reward == 1 or reward == -5:
                    done = True
                state = next_state
            rewards.append(total_reward)
        return rewards

# GPU Agent
class AgentGPU:
    def __init__(self):
        self.Q = np.zeros((BOARD_ROWS, BOARD_COLS, 4), dtype=np.float32)
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.q_flat = self.Q.flatten().copy()
        self.Q_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.q_flat)
        self.move_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=move.flatten())

    def choose_action(self, state):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.Q[state[0], state[1], :])
        else:
            return np.random.choice(actions)

    def get_reward(self, state):
        if tuple(state) == (4, 4):
            return 1
        if tuple(state) in [(1, 0), (3, 1), (4, 2), (1, 3)]:
            return -5
        return -1

    def train(self, episodes=10000):
        rewards = []
        for _ in range(episodes):
            state = np.array([0, 0], dtype=np.int32)
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state = state + move[action]
                next_state = np.clip(next_state, [0, 0], [BOARD_ROWS-1, BOARD_COLS-1])
                reward = self.get_reward(next_state)
                total_reward += reward

                # Launch the kernel without reallocating buffers every step
                program.update_q(
                    queue,
                    (1,),
                    None,
                    self.Q_buf,
                    self.move_buf,
                    np.int32(BOARD_ROWS),
                    np.int32(BOARD_COLS),
                    np.float32(self.alpha),
                    np.float32(self.gamma),
                    np.int32(action),
                    np.int32(state[0]),
                    np.int32(state[1]),
                    np.int32(reward)
                )

                queue.finish()

                if reward == 1 or reward == -5:
                    done = True
                state = next_state

            rewards.append(total_reward)

        # Final copy back once after all training
        cl.enqueue_copy(queue, self.q_flat, self.Q_buf)
        queue.finish()
        self.Q = self.q_flat.reshape((BOARD_ROWS, BOARD_COLS, 4))
        return rewards

if __name__ == "__main__":
    print("Training CPU Agent...")
    agent_cpu = AgentCPU()
    start_cpu = time.time()
    rewards_cpu = agent_cpu.train(episodes)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    print("Training GPU Agent (OpenCL)...")
    agent_gpu = AgentGPU()
    start_gpu = time.time()
    rewards_gpu = agent_gpu.train(episodes)
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    speedup = cpu_time / gpu_time

    print(f"CPU Training Time: {cpu_time:.2f} seconds")
    print(f"GPU Training Time: {gpu_time:.2f} seconds")
    print(f"Speedup: {speedup:.2f}x")

    plt.plot(rewards_cpu, label='CPU Rewards')
    plt.plot(rewards_gpu, label='GPU Rewards', linestyle='dashed')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('CPU vs GPU (OpenCL) Training Rewards')
    plt.legend()
    plt.grid()
    plt.show()