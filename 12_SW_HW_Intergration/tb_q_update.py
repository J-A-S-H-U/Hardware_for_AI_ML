import cocotb
from cocotb.triggers import RisingEdge
import struct

# Convert float <-> 32-bit binary representation
def float_to_bits(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]

def bits_to_float(b):
    return struct.unpack('<f', struct.pack('<I', b))[0]

@cocotb.test()
async def q_update_test(dut):
    q      = 0.5
    reward = -1
    max_q  = 0.8
    alpha  = 0.5
    gamma  = 0.9

    dut.q_val.value     = float_to_bits(q)
    dut.reward.value    = float_to_bits(reward)
    dut.max_q_next.value = float_to_bits(max_q)
    dut.alpha.value     = float_to_bits(alpha)
    dut.gamma.value     = float_to_bits(gamma)

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    result = bits_to_float(dut.q_updated.value.integer)
    print(f"Hardware Q updated: {result}")