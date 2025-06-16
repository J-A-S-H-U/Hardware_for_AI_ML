module barrel_shifter_32 (
    input  [31:0] data_in,     // Input data
    input  [4:0]  shift_amt,   // Shift amount (0–31)
    input  [1:0]  shift_type,  // 00 = LSL, 01 = LSR, 10 = ASR, 11 = ROR
    output reg [31:0] data_out
);

    always @(*) begin
        case (shift_type)
            2'b00: // Logical Shift Left (LSL)
                data_out = data_in << shift_amt;
            2'b01: // Logical Shift Right (LSR)
                data_out = data_in >> shift_amt;
            2'b10: // Arithmetic Shift Right (ASR)
                data_out = $signed(data_in) >>> shift_amt;
            2'b11: // Rotate Right (ROR)
                data_out = (data_in >> shift_amt) | (data_in << (32 - shift_amt));
            default:
                data_out = 32'b0;
        endcase
    end

endmodule
