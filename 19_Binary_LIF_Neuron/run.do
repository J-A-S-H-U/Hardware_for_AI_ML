vlib work

vlog +acc binary_lif_neuron.sv
vlog +acc tb_binary_lif_neuron.sv

vsim work.tb_binary_lif_neuron

do wave.do

run -all

add log -r /*
