onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /tb_binary_lif_neuron/uut/THRESHOLD
add wave -noupdate /tb_binary_lif_neuron/uut/LEAK_FACTOR_NUM
add wave -noupdate /tb_binary_lif_neuron/uut/LEAK_FACTOR_DEN
add wave -noupdate -group Interface /tb_binary_lif_neuron/uut/clk
add wave -noupdate -group Interface /tb_binary_lif_neuron/uut/rst
add wave -noupdate -group Interface /tb_binary_lif_neuron/uut/I
add wave -noupdate -group Interface /tb_binary_lif_neuron/uut/S
add wave -noupdate -group Interface /tb_binary_lif_neuron/uut/P
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {0 ps} 0}
quietly wave cursor active 0
configure wave -namecolwidth 298
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 0
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ps
update
WaveRestoreZoom {0 ps} {8321 ps}
