`timescale 1ns/1ps

module biquad_controller #(
    parameter int CHANNELS = 32,
    parameter int CHANNEL_BITS = (CHANNELS > 1 ? $clog2(CHANNELS) : 1)
)(
    input logic clk,
    input logic rst_n,

    input logic in_valid,
    output logic in_ready,
    input logic signed [23:0] in_sample,
    input logic [CHANNEL_BITS-1:0] in_channel,

    output logic out_valid,
    input logic out_ready,
    output logic signed [23:0] out_sample,
    output logic [CHANNEL_BITS-1:0] out_channel,

    input logic signed [31:0] b0, b1, b2, a1, a2
);

typedef enum logic [1:0] {
    IDLE,
    COMPUTE,
    OUTPUT
} state_t;

state_t state, next_state;
assign in_ready = (state == IDLE);
assign out_valid = (state == OUTPUT);

logic signed [23:0] held_sample;
logic [CHANNEL_BITS-1:0] held_channel;

logic signed [23:0] x1, x2, y1, y2;
logic signed [23:0] new_y;


biquad_state #(
    .CHANNELS(CHANNELS),
    .CHANNEL_BITS(CHANNEL_BITS)

) state_storage (
    .clk(clk),
    .rst_n(rst_n),
    .read_channel(held_channel),
    .x1(x1),
    .x2(x2),
    .y1(y1),
    .y2(y2),

    .write_enable(state == COMPUTE),
    .write_channel(held_channel),
    .new_x(held_sample),
    .new_y(new_y)
);

biquad_arithmetic arithmetic_unit(
    .x0(held_sample),
    .x1(x1),
    .x2(x2),
    .y1(y1),
    .y2(y2),
    .b0(b0),
    .b1(b1),
    .b2(b2),
    .a1(a1),
    .a2(a2),
    .new_y(new_y)
);



always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        held_sample <= 0;
        held_channel <= 0;
        out_sample <= '0;
        out_channel <= '0;
    end else begin
        state <= next_state;
        if (in_valid && in_ready) begin
            held_sample <= in_sample;
            held_channel <= in_channel;
        end
        if (state == COMPUTE) begin
            out_sample <= new_y;
            out_channel <= held_channel;
        end
    end
end

always_comb
begin
    next_state = state;
    case (state)
        IDLE: if (in_valid && in_ready) next_state = COMPUTE;
        COMPUTE: next_state = OUTPUT;
        OUTPUT: if (out_valid && out_ready) next_state = IDLE;
        default : next_state = IDLE;
    endcase
end

endmodule
