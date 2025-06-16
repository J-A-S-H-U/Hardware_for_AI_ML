import numpy as np
import random
import matplotlib.pyplot as plt
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from cocotb.binary import BinaryValue
import os
import asyncio

# Hardware interface class
class QUpdateHardware:
    def __init__(self, dut):
        self.dut = dut
        self.clk = dut.clk
        self.rst = dut.rst
        self.q_sa = dut.q_sa
        self.reward = dut.reward
        self.max_q_spap = dut.max_q_spap
        self.alpha = dut.alpha
        self.gamma = dut.gamma
        self.q_new = dut.q_new
        
    async def initialize(self):
        cocotb.start_soon(Clock(self.clk, 10, units="ns").start())
        self.rst.value = 1
        await RisingEdge(self.clk)
        await RisingEdge(self.clk)
        self.rst.value = 0
        await RisingEdge(self.clk)
        
    async def compute_q_update(self, q_sa, reward, max_q_spap, alpha, gamma):
        # Convert float values to Q4.12 fixed-point
        def float_to_fixed(x):
            return int(x * (1 << 12))
            
        def fixed_to_float(x):
            return x / (1 << 12)
            
        # Set input values
        self.q_sa.value = float_to_fixed(q_sa)
        self.reward.value = float_to_fixed(reward)
        self.max_q_spap.value = float_to_fixed(max_q_spap)
        self.alpha.value = float_to_fixed(alpha)
        self.gamma.value = float_to_fixed(gamma)
        
        # Wait for pipeline to fill (5 clock cycles for 5 stages)
        for _ in range(5):
            await RisingEdge(self.clk)
            
        # Get result and convert back to float
        return fixed_to_float(self.q_new.value.integer)

# Main Agent class with hardware acceleration
class AgentHW:
    def __init__(self, dut=None, use_hardware=False):
        self.states = []
        self.actions = [0,1,2,3]    # up, down, left, right
        self.State = State()
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.isEnd = self.State.isEnd
        self.plot_reward = []
        self.Q = {}
        self.new_Q = {}
        self.rewards = 0
        self.use_hardware = use_hardware
        self.hw_interface = None
        
        if use_hardware and dut is not None:
            self.hw_interface = QUpdateHardware(dut)
            
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for k in range(len(self.actions)):
                    self.Q[(i, j, k)] = 0
                    self.new_Q[(i, j, k)] = 0

    async def initialize_hardware(self):
        if self.hw_interface:
            await self.hw_interface.initialize()

    def Action(self):
        rnd = random.random()
        mx_nxt_reward = -10
        action = None
        
        if rnd > self.epsilon:
            for k in self.actions:
                i,j = self.State.state
                nxt_reward = self.Q[(i,j, k)]
                if nxt_reward >= mx_nxt_reward:
                    action = k
                    mx_nxt_reward = nxt_reward
        else:
            action = np.random.choice(self.actions)
            
        position = self.State.nxtPosition(action)
        return position, action
    
    async def Q_Learning(self, episodes):
        x = 0
        while x < episodes:
            if self.isEnd:
                reward = self.State.getReward()
                self.rewards += reward
                self.plot_reward.append(self.rewards)
                
                i,j = self.State.state
                for a in self.actions:
                    self.new_Q[(i,j,a)] = round(reward,3)
                    
                self.State = State()
                self.isEnd = self.State.isEnd
                self.rewards = 0
                x += 1
            else:
                mx_nxt_value = -10
                next_state, action = self.Action()
                i,j = self.State.state
                reward = self.State.getReward()
                self.rewards += reward
                
                # Get max Q(s',a')
                max_q_spap = -10
                for a in self.actions:
                    nxtStateAction = (next_state[0], next_state[1], a)
                    q_value = self.Q[nxtStateAction]
                    if q_value > max_q_spap:
                        max_q_spap = q_value
                
                # Compute Q update - use hardware if enabled
                if self.use_hardware and self.hw_interface:
                    q_update = await self.hw_interface.compute_q_update(
                        q_sa=self.Q[(i,j,action)],
                        reward=reward,
                        max_q_spap=max_q_spap,
                        alpha=self.alpha,
                        gamma=self.gamma
                    )
                else:
                    # Software fallback
                    q_update = (1-self.alpha)*self.Q[(i,j,action)] + \
                              self.alpha*(reward + self.gamma*max_q_spap)
                
                self.new_Q[(i,j,action)] = round(q_update, 3)
                
                self.State = State(state=next_state)
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd
            
            self.Q = self.new_Q.copy()
        
    def plot(self, episodes):
        plt.plot(self.plot_reward)
        plt.show()
        
    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('-----------------------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                mx_nxt_value = -10
                for a in self.actions:
                    nxt_value = self.Q[(i,j,a)]
                    if nxt_value >= mx_nxt_value:
                        mx_nxt_value = nxt_value
                out += str(mx_nxt_value).ljust(6) + ' | '
            print(out)
        print('-----------------------------------------------')