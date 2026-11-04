`timescale 1ns/1ps
`default_nettype none

// Stand-in for rtl/biquad_arithmetic.sv used only by formal/biquad_controller.sby.
// new_y is an unconstrained sequence, so the controller proof checks that the
// arithmetic result is registered and written into the addressed delay line.
// It does not check the Q2.30 multiply-accumulate. That stays with
// rtl/tb/biquad_arithmetic_tb.sv.
//
// read_slang drives (* anyseq *) with a constant X. formal/biquad_controller.sby
// deletes that buffer and attaches one $anyseq cell before prep.

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
    (* anyseq *) logic signed [23:0] free_y;
    assign new_y = free_y;
endmodule

`default_nettype wire
