module binary_lif_neuron (
    input logic clk,
    input logic rst,             // Active-high synchronous reset
    input logic I,               // Binary input (0 or 1)
    output logic S               // Output spike
);
    // Parameters
    parameter int THRESHOLD = 10;            // θ: Threshold for firing
    parameter int LEAK_FACTOR_NUM = 9;       // λ numerator (e.g., 0.9 → 9/10)
    parameter int LEAK_FACTOR_DEN = 10;      // λ denominator

    // Internal state
    integer P = 0;                            // Membrane potential

    always_ff @(posedge clk) begin
        if (rst) begin
            P <= 0;
            S <= 0;
        end else begin
            // Apply leak and accumulate input
            P <= (P * LEAK_FACTOR_NUM) / LEAK_FACTOR_DEN + I;

            // Threshold check
            if (P >= THRESHOLD) begin
                S <= 1;
                P <= 0;  // Reset potential after spike
            end else begin
                S <= 0;
            end
        end
    end
endmodule
