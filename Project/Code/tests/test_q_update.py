import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from cocotb.binary import BinaryValue
import random
import numpy as np

from frozen_lake_hw import QUpdateHardware

@cocotb.test()
async def test_q_update(dut):
    # Initialize hardware interface
    hw = QUpdateHardware(dut)
    await hw.initialize()
    
    # Test parameters
    alpha = 0.5
    gamma = 0.9
    
    # Run random tests
    for _ in range(100):
        q_sa = random.uniform(-10, 10)
        reward = random.uniform(-5, 5)
        max_q_spap = random.uniform(-10, 10)
        
        # Compute expected result (software)
        expected = (1-alpha)*q_sa + alpha*(reward + gamma*max_q_spap)
        
        # Compute hardware result
        hw_result = await hw.compute_q_update(q_sa, reward, max_q_spap, alpha, gamma)
        
        # Compare with tolerance for floating point
        assert abs(hw_result - expected) < 0.01, \
            f"HW result {hw_result} doesn't match SW result {expected} for " \
            f"inputs q_sa={q_sa}, reward={reward}, max_q_spap={max_q_spap}"
    
    dut._log.info("All tests passed!")