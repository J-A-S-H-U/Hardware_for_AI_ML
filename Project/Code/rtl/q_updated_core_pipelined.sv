module q_update_core_pipelined #(
  parameter WIDTH = 32
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire [WIDTH-1:0]     q_sa,        // Q(s,a)
  input  wire [WIDTH-1:0]     reward,      // r
  input  wire [WIDTH-1:0]     max_q_spap,  // max Q(s',a')
  input  wire [WIDTH-1:0]     alpha,       // α
  input  wire [WIDTH-1:0]     gamma,       // γ
  output reg  [WIDTH-1:0]     q_new        // Updated Q(s,a)
);

  // constants
  localparam FIXED_ONE = {4'h1,12'h000}; // 1.0 in Q4.12

  // Wires for raw operations
  wire [2*WIDTH-1:0] mul1_out;
  wire [WIDTH-1:0]   gamma_term;
  wire [WIDTH-1:0]   reward_plus;
  wire [2*WIDTH-1:0] mul2_out;
  wire [WIDTH-1:0]   delta_term;
  wire [2*WIDTH-1:0] mul3_out;
  wire [WIDTH-1:0]   qsa_term;
  wire [WIDTH-1:0]   sum_out;
  wire [WIDTH-1:0]   one_minus_alpha;

  // Stage 0: compute one_minus_alpha (combinational)
  brent_kung_subtractor #(WIDTH) sub_alpha (
    .a(FIXED_ONE), .b(alpha), .result(one_minus_alpha)
  );

  // Stage 1: gamma*max_q  → shift
  wallace_multiplier #(WIDTH) mul1 (
    .a(gamma), .b(max_q_spap), .product(mul1_out)
  );
  barrel_shifter #(2*WIDTH) shift1 (
    .data_in(mul1_out), .shift_amt(12), .data_out(gamma_term)
  );
  reg [WIDTH-1:0]  reg_stage1; // pipeline register
  always @(posedge clk or posedge rst) begin
    if (rst) reg_stage1 <= 0;
    else     reg_stage1 <= gamma_term;
  end

  // Stage 2: add reward + reg_stage1  → mul2  → shift2
  brent_kung_adder #(WIDTH) add1 (
    .a(reward), .b(reg_stage1), .sum(reward_plus)
  );
  wallace_multiplier #(WIDTH) mul2 (
    .a(alpha), .b(reward_plus), .product(mul2_out)
  );
  barrel_shifter #(2*WIDTH) shift2 (
    .data_in(mul2_out), .shift_amt(12), .data_out(delta_term)
  );
  reg [WIDTH-1:0]  reg_stage2;
  always @(posedge clk or posedge rst) begin
    if (rst) reg_stage2 <= 0;
    else     reg_stage2 <= delta_term;
  end

  // Stage 3: one_minus_alpha * q_sa → shift3
  wallace_multiplier #(WIDTH) mul3 (
    .a(one_minus_alpha), .b(q_sa), .product(mul3_out)
  );
  barrel_shifter #(2*WIDTH) shift3 (
    .data_in(mul3_out), .shift_amt(12), .data_out(qsa_term)
  );
  reg [WIDTH-1:0]  reg_stage3;
  always @(posedge clk or posedge rst) begin
    if (rst) reg_stage3 <= 0;
    else     reg_stage3 <= qsa_term;
  end

  // Stage 4: final add → output register
  brent_kung_adder #(WIDTH) add2 (
    .a(reg_stage3), .b(reg_stage2), .sum(sum_out)
  );
  always @(posedge clk or posedge rst) begin
    if (rst) q_new <= 0;
    else     q_new <= sum_out;
  end

endmodule
