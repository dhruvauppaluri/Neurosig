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

always_ff @(posedge clk or negedge rst_n)
begin
    if (!rst_n)
    begin
        for (int i = 0; i < CHANNELS; i++) begin
            x1_mem[i] <= 0;
            x2_mem[i] <= 0;
            y1_mem[i] <= 0;
            y2_mem[i] <= 0;
        end
    end
    else
    begin
        if (write_enable)
        begin
            x1_mem[write_channel] <= new_x;
            x2_mem[write_channel] <= x1_mem[write_channel];
            y1_mem[write_channel] <= new_y;
            y2_mem[write_channel] <= y1_mem[write_channel];
        end
    end
end
endmodule
