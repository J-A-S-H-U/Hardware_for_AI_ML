`timescale 1ns/1ps

module tb;

    // DUT I/O
    reg  [31:0] A, B;
    reg         Cin;
    wire [31:0] Sum, Diff;
    wire        Cout, Bout;
    wire [63:0] Product;

    reg  [31:0] data_in;
    reg  [4:0]  shift_amt;
    reg  [1:0]  shift_type;
    wire [31:0] data_out;

    // Golden model outputs
    reg  [31:0] expected_sum, expected_diff, expected_shift;
    reg  [63:0] expected_product;
    reg         expected_cout, expected_bout;

    // DUTs
    brent_kung_adder_32 uut_adder (
        .A(A),
        .B(B),
        .Cin(Cin),
        .Sum(Sum),
        .Cout(Cout)
    );

    brent_kung_subtractor_32 uut_sub (
        .A(A),
        .B(B),
        .Diff(Diff),
        .Bout(Bout)
    );

    barrel_shifter_32 uut_shift (
        .data_in(data_in),
        .shift_amt(shift_amt),
        .shift_type(shift_type),
        .data_out(data_out)
    );

    wallace_multiplier_32 uut_mult (
        .A(A),
        .B(B),
        .P(Product)
    );

    // Tasks
    task check_adder;
        input [31:0] A, B;
        input        Cin;
        begin
            {expected_cout, expected_sum} = A + B + Cin;
            if (Sum !== expected_sum || Cout !== expected_cout)
                $display("Adder MISMATCH: A=%0d, B=%0d, Cin=%0b | Sum=%0d (exp %0d), Cout=%0b (exp %0b)", A, B, Cin, Sum, expected_sum, Cout, expected_cout);
            else
                $display("Adder PASS: A=%0d, B=%0d, Sum=%0d", A, B, Sum);
        end
    endtask

    task check_subtractor;
        input [31:0] A, B;
        begin
            expected_diff = A - B;
            expected_bout = (A < B);
            if (Diff !== expected_diff || Bout !== expected_bout)
                $display("Subtractor MISMATCH: A=%0d, B=%0d | Diff=%0d (exp %0d), Bout=%0b (exp %0b)", A, B, Diff, expected_diff, Bout, expected_bout);
            else
                $display("Subtractor PASS: A=%0d, B=%0d, Diff=%0d", A, B, Diff);
        end
    endtask

    task check_shifter;
        input [31:0] din;
        input [4:0]  shamt;
        input [1:0]  stype;
        begin
            case (stype)
                2'b00: expected_shift = din << shamt;
                2'b01: expected_shift = din >> shamt;
                2'b10: expected_shift = $signed(din) >>> shamt;
                2'b11: expected_shift = (din >> shamt) | (din << (32 - shamt));
                default: expected_shift = 32'hx;
            endcase

            if (data_out !== expected_shift)
                $display("Shifter MISMATCH: Type=%b, In=%h, Amt=%0d | Out=%h (exp %h)", stype, din, shamt, data_out, expected_shift);
            else
                $display("Shifter PASS: Type=%b, In=%h, Out=%h", stype, din, data_out);
        end
    endtask

    task check_multiplier;
        input [31:0] A, B;
        begin
            expected_product = A * B;
            if (Product !== expected_product)
                $display("Multiplier MISMATCH: A=%d, B=%d | Product=%h (exp %h)", A, B, Product, expected_product);
            else
                $display("Multiplier PASS: A=%d, B=%d, Product=%h", A, B, Product);
        end
    endtask

    // Test sequence
    initial begin
        $display("\n===== Brent-Kung Adder Tests =====");
        A = 32'd12345; B = 32'd67890; Cin = 0; #5 check_adder(A, B, Cin);
        A = 32'hFFFF_FFFF; B = 32'd1; Cin = 0; #5 check_adder(A, B, Cin);

        $display("\n===== Brent-Kung Subtractor Tests =====");
        A = 32'd500; B = 32'd300; #5 check_subtractor(A, B);
        A = 32'd100; B = 32'd200; #5 check_subtractor(A, B);

        $display("\n===== Barrel Shifter Tests =====");
        data_in = 32'hF0F0F0F0; shift_amt = 5'd4;

        shift_type = 2'b00; #5 check_shifter(data_in, shift_amt, shift_type); // LSL
        shift_type = 2'b01; #5 check_shifter(data_in, shift_amt, shift_type); // LSR
        shift_type = 2'b10; #5 check_shifter(data_in, shift_amt, shift_type); // ASR
        shift_type = 2'b11; #5 check_shifter(data_in, shift_amt, shift_type); // ROR

        $display("\n===== Wallace Tree Multiplier Tests =====");
        A = 32'd13; B = 32'd7; #5 check_multiplier(A, B);
        A = 32'd65535; B = 32'd2; #5 check_multiplier(A, B);
        A = 32'd123456; B = 32'd789; #5 check_multiplier(A, B);

        $display("\n===== All Tests Completed =====");
        $finish;
    end

endmodule
