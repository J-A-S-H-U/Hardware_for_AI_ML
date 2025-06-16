TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(shell pwd)/../rtl/q_update_core_pipelined.v \
                  $(shell pwd)/../rtl/brent_kung_adder.v \
                  $(shell pwd)/../rtl/brent_kung_subtractor.v \
                  $(shell pwd)/../rtl/wallace_multiplier.v \
                  $(shell pwd)/../rtl/barrel_shifter.v

TOPLEVEL = q_update_core_pipelined
MODULE = test_q_update

include $(shell cocotb-config --makefiles)/Makefile.sim