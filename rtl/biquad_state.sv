`timescale 1ns/1ps

module biquad_state #(
    parameter int CHANNELS = 32,
    parameter int CHANNEL_BITS = (CHANNELS  > 1 ? $clog2(CHANNELS) : 1)
) (
    input  logic clk,
    input  logic rst_n,
    input  logic [CHANNEL_BITS-1:0] read_channel,
    output logic signed [23:0] x1,
    output logic signed [23:0] x2,
    output logic signed [23:0] y1,
    output logic signed [23:0] y2,

    input logic write_enable,
    input logic [CHANNEL_BITS-1:0] write_channel,
    input logic signed [23:0] new_x,
    input logic signed [23:0] new_y

);

logic signed [23:0] x1_mem [0:CHANNELS-1];
logic signed [23:0] x2_mem [0:CHANNELS-1];
logic signed [23:0] y1_mem [0:CHANNELS-1];
logic signed [23:0] y2_mem [0:CHANNELS-1]; // actual per channel memory

assign x1 = x1_mem[read_channel];
assign x2 = x2_mem[read_channel];
assign y1 = y1_mem[read_channel];
assign y2 = y2_mem[read_channel];

genvar c;
generate for (c = 0; c < CHANNELS; c = c + 1) begin : channel_state
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x1_mem[c] <= 0;
            x2_mem[c] <= 0;
            y1_mem[c] <= 0;
            y2_mem[c] <= 0;
        end else if (write_enable && write_channel == CHANNEL_BITS'(c)) begin
            x1_mem[c] <= new_x;
            x2_mem[c] <= x1_mem[c];
            y1_mem[c] <= new_y;
            y2_mem[c] <= y1_mem[c];
        end
    end
end endgenerate
endmodule
