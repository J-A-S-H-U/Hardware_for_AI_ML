module QValueMaxFinder #(
    parameter WIDTH = 16,     // Bit-width of Q-values (fixed-point representation)
    parameter ACTIONS = 4     // Number of possible actions
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic [WIDTH-1:0]       Q_values [0:ACTIONS-1], // Q-values for each action
    input  logic [WIDTH-1:0]       reward,
    input  logic [WIDTH-1:0]       alpha,                  // Learning rate (scaled appropriately)
    input  logic [WIDTH-1:0]       gamma,                  // Discount factor (scaled appropriately)
    output logic [WIDTH-1:0]       max_Q_value             // Output: maximum updated Q-value
);

    logic [WIDTH-1:0] updated_Q [0:ACTIONS-1];
    logic [WIDTH-1:0] temp_max;
    integer i;

    // Update Q-values based on Bellman equation
    always_comb begin
        for (i = 0; i < ACTIONS; i++) begin
            // Bellman update: Q' = (1-alpha)*Q + alpha*(reward + gamma*Q)
            updated_Q[i] = (({1'b0, {(WIDTH-1){1'b1}}} - alpha) * Q_values[i] +
                           (alpha * (reward + (gamma * Q_values[i] >> (WIDTH/2))))) >> (WIDTH/2);
        end
    end

    // Find the maximum updated Q-value
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_Q_value <= 0;
        end else begin
            temp_max = updated_Q[0];
            for (i = 1; i < ACTIONS; i++) begin
                if (updated_Q[i] > temp_max) begin
                    temp_max = updated_Q[i];
                end
            end
            max_Q_value <= temp_max;
        end
    end

endmodule
