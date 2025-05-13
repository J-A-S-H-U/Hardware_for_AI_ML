`timescale 1ns/1ns

module tb_binary_lif_neuron;

    logic clk;
    logic rst;
    logic I;
    logic S;

    // Instantiate the neuron
    binary_lif_neuron #(.THRESHOLD(5)) uut (
        .clk(clk),
        .rst(rst),
        .I(I),
        .S(S)
    );

    // Clock generation: 10ns period
    always #5 clk = ~clk;

    // Simulation logic
    initial begin
        $display("Time\tI\tS");

        // Initialize
        clk = 0;
        rst = 1; I = 0;
        #10 rst = 0;

        // 1. Constant input below threshold (0s)
        repeat (5) begin
            I = 0; #10;
            $display("%0dns\t%b\t%b", $time, I, S);
        end

        // 2. Accumulating input (I = 1 over time)
        repeat (10) begin
            I = 1; #10;
            $display("%0dns\t%b\t%b", $time, I, S);
        end

        // 3. No input - observe leakage
        repeat (10) begin
            I = 0; #10;
            $display("%0dns\t%b\t%b", $time, I, S);
        end

        // 4. Strong input burst to cause spike
        repeat (3) begin
            I = 1; #10;
            $display("%0dns\t%b\t%b", $time, I, S);
        end

        $finish;
    end



endmodule
