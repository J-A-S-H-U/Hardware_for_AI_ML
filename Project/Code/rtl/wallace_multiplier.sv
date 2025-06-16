module wallace_multiplier_32 (
    input  [31:0] A,
    input  [31:0] B,
    output [63:0] P
);

    wire [31:0] partial_products[31:0];
    genvar i;

    // Generate partial products
    generate
        for (i = 0; i < 32; i = i + 1) begin : gen_partial_products
            assign partial_products[i] = A & {32{B[i]}};
        end
    endgenerate

    wire [63:0] sum_stage[31:0];

    generate
        for (i = 0; i < 32; i = i + 1) begin : gen_shifted_partial
            assign sum_stage[i] = { {32{i'b0}}, partial_products[i] } << i;
        end
    endgenerate

    // Accumulate all shifted partials using adder tree
    function [63:0] sum_tree;
        input integer n;
        input [63:0] arr[];
        integer i;
        begin
            if (n == 1) begin
                sum_tree = arr[0];
            end else begin
                integer new_size = (n + 1) >> 1;
                reg [63:0] temp[0:31];
                for (i = 0; i < new_size; i = i + 1) begin
                    if ((2*i+1) < n)
                        temp[i] = arr[2*i] + arr[2*i+1];
                    else
                        temp[i] = arr[2*i];
                end
                sum_tree = sum_tree(new_size, temp);
            end
        end
    endfunction

    assign P = sum_tree(32, sum_stage);

endmodule
