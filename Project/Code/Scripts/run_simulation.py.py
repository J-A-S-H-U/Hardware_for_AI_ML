import asyncio
from frozen_lake_hw import AgentHW
from frozen_lake_sw import Agent
import matplotlib.pyplot as plt
import cocotb
from cocotb.simulator import Simulator
import os

async def run_hardware_simulation(dut):
    # Create hardware-accelerated agent
    hw_agent = AgentHW(dut=dut, use_hardware=True)
    await hw_agent.initialize_hardware()
    
    # Run Q-learning with hardware acceleration
    await hw_agent.Q_Learning(episodes=10000)
    
    # Plot results
    hw_agent.plot(episodes=10000)
    hw_agent.showValues()
    
    return hw_agent.plot_reward

def run_software_simulation():
    # Create software-only agent
    sw_agent = Agent()
    
    # Run Q-learning in software
    sw_agent.Q_Learning(episodes=10000)
    
    # Plot results
    sw_agent.plot(episodes=10000)
    sw_agent.showValues()
    
    return sw_agent.plot_reward

async def main():
    # Run software version
    print("Running software implementation...")
    sw_rewards = run_software_simulation()
    
    # Run hardware-accelerated version
    print("\nRunning hardware-accelerated implementation...")
    hw_rewards = await cocotb.start(run_hardware_simulation(dut))
    
    # Compare results
    plt.figure(figsize=(10, 6))
    plt.plot(sw_rewards, label='Software')
    plt.plot(hw_rewards, label='Hardware Accelerated')
    plt.title('Comparison of Software vs Hardware-Accelerated Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())