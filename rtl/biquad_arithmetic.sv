`timescale 1ns/1ps

module biquad_arithmetic (
    input logic signed [23:0] x0,
    input logic signed [23:0] x1,
    input logic signed [23:0] x2,
    input logic signed [23:0] y1,
    input logic signed [23:0] y2,

    input logic signed [31:0] b0,
    input logic signed [31:0] b1,
    input logic signed [31:0] b2,
    input logic signed [31:0] a1,
    input logic signed [31:0] a2,

    output logic signed [23:0] new_y
);

logic signed [63:0] accumulator;

logic signed [55:0] mult1;
logic signed [55:0] mult2;
logic signed [55:0] mult3;
logic signed [55:0] mult4;
logic signed [55:0] mult5;

logic signed [63:0] reduced;

always_comb begin
    mult1 = b0 * x0;
    mult2 = b1 * x1;
    mult3 = b2 * x2;
    mult4 = a1 * y1;
    mult5 = a2 * y2;
    accumulator = 64'(mult1) + 64'(mult2) + 64'(mult3)
                - 64'(mult4) - 64'(mult5);

    if (accumulator >= 0) reduced = accumulator >>> 30;
    else reduced = -((-accumulator) >>> 30);

    if (reduced > 64'sd8388607) new_y = 24'sd8388607;
    else if (reduced < -64'sd8388608) new_y = 24'sh800000;
    else new_y = reduced[23:0];
end




endmodule
