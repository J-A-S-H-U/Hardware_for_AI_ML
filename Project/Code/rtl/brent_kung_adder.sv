module brent_kung_adder_32 (
    input  [31:0] A,
    input  [31:0] B,
    input         Cin,
    output [31:0] Sum,
    output        Cout
);

    wire [31:0] G, P;       // Generate and Propagate
    wire [31:0] C;          // Carry

    assign G = A & B;
    assign P = A ^ B;

    // Level 0
    wire [31:0] G0 = G;
    wire [31:0] P0 = P;

    // Level 1: (1 bit groups)
    wire [31:0] G1, P1;
    genvar i;
    generate
        for (i = 1; i < 32; i = i + 2) begin
            assign G1[i] = G0[i] | (P0[i] & G0[i-1]);
            assign P1[i] = P0[i] & P0[i-1];
        end
        for (i = 0; i < 32; i = i + 2) begin
            assign G1[i] = G0[i];
            assign P1[i] = P0[i];
        end
    endgenerate

    // Level 2: (2 bit groups)
    wire [31:0] G2, P2;
    generate
        for (i = 3; i < 32; i = i + 4) begin
            assign G2[i] = G1[i] | (P1[i] & G1[i-2]);
            assign P2[i] = P1[i] & P1[i-2];
        end
        for (i = 0; i < 32; i = i + 1) begin
            if (i % 4 != 3) begin
                assign G2[i] = G1[i];
                assign P2[i] = P1[i];
            end
        end
    endgenerate

    // Level 3: (4 bit groups)
    wire [31:0] G3, P3;
    generate
        for (i = 7; i < 32; i = i + 8) begin
            assign G3[i] = G2[i] | (P2[i] & G2[i-4]);
            assign P3[i] = P2[i] & P2[i-4];
        end
        for (i = 0; i < 32; i = i + 1) begin
            if (i % 8 != 7) begin
                assign G3[i] = G2[i];
                assign P3[i] = P2[i];
            end
        end
    endgenerate

    // Level 4: (8 bit groups)
    wire [31:0] G4;
    generate
        for (i = 15; i < 32; i = i + 16) begin
            assign G4[i] = G3[i] | (P3[i] & G3[i-8]);
        end
        for (i = 0; i < 32; i = i + 1) begin
            if (i % 16 != 15) begin
                assign G4[i] = G3[i];
            end
        end
    endgenerate

    // Backward propagation to fill carries
    assign C[0] = Cin;

    generate
        for (i = 1; i < 32; i = i + 1) begin : carry_gen
            assign C[i] = G4[i-1] | (P0[i-1] & C[i-1]);
        end
    endgenerate

    // Sum and Cout
    assign Sum = P0 ^ C;
    assign Cout = G4[31] | (P0[31] & C[31]);

endmodule
