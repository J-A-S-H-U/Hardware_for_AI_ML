module brent_kung_subtractor_32 (
    input  [31:0] A,      // Minuend
    input  [31:0] B,      // Subtrahend
    output [31:0] Diff,   // Difference (A - B)
    output        Bout    // Final borrow (1 if A < B)
);

    wire [31:0] B_compl;
    wire [31:0] Sum;
    wire        Cout;

    // Two's complement of B → (~B + 1)
    assign B_compl = ~B;

    // Brent-Kung Adder: A + (~B + 1) = A - B
    brent_kung_adder_32 adder (
        .A(A),
        .B(B_compl),
        .Cin(1'b1),        // Adding 1 for two's complement
        .Sum(Sum),
        .Cout(Cout)
    );

    assign Diff = Sum;
    assign Bout = ~Cout;   // If carry-out is 0, then borrow occurred

endmodule
