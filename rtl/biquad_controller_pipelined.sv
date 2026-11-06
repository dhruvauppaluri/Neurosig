`timescale 1ns/1ps

// Timing-oriented schedule of the same Q2.30 / signed-24 biquad arithmetic.
// Each accepted sample completes before another is accepted by this section.
module biquad_controller_pipelined #(
    parameter int CHANNELS = 32,
    parameter int CHANNEL_BITS = (CHANNELS > 1 ? $clog2(CHANNELS) : 1)
) (
    input logic clk, rst_n,
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
    typedef enum logic [2:0] {IDLE, LOAD, MULT, ACCUM, REDUCE, OUTPUT} state_t;
    state_t state;
    logic signed [23:0] held_sample;
    logic [CHANNEL_BITS-1:0] held_channel;
    logic signed [23:0] x1, x2, y1, y2;
    logic signed [23:0] x1_reg, x2_reg, y1_reg, y2_reg;
    logic signed [55:0] mult1, mult2, mult3, mult4, mult5;
    logic signed [63:0] accumulator, reduced;
    logic signed [23:0] new_y;

    assign in_ready = state == IDLE;
    assign out_valid = state == OUTPUT;
    always_comb begin
        if (accumulator >= 0) reduced = accumulator >>> 30;
        else reduced = -((-accumulator) >>> 30);
        if (reduced > 64'sd8388607) new_y = 24'sd8388607;
        else if (reduced < -64'sd8388608) new_y = 24'sh800000;
        else new_y = reduced[23:0];
    end

    biquad_state #(.CHANNELS(CHANNELS), .CHANNEL_BITS(CHANNEL_BITS)) state_storage (
        .clk(clk), .rst_n(rst_n), .read_channel(held_channel),
        .x1(x1), .x2(x2), .y1(y1), .y2(y2),
        .write_enable(state == REDUCE), .write_channel(held_channel),
        .new_x(held_sample), .new_y(new_y)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            held_sample <= 0;
            held_channel <= 0;
            x1_reg <= 0; x2_reg <= 0; y1_reg <= 0; y2_reg <= 0;
            mult1 <= 0; mult2 <= 0; mult3 <= 0; mult4 <= 0; mult5 <= 0;
            accumulator <= 0;
            out_sample <= 0;
            out_channel <= 0;
        end else begin
            case (state)
                IDLE: if (in_valid) begin
                    held_sample <= in_sample;
                    held_channel <= in_channel;
                    state <= LOAD;
                end
                LOAD: begin
                    x1_reg <= x1; x2_reg <= x2; y1_reg <= y1; y2_reg <= y2;
                    state <= MULT;
                end
                MULT: begin
                    mult1 <= b0 * held_sample;
                    mult2 <= b1 * x1_reg;
                    mult3 <= b2 * x2_reg;
                    mult4 <= a1 * y1_reg;
                    mult5 <= a2 * y2_reg;
                    state <= ACCUM;
                end
                ACCUM: begin
                    accumulator <= 64'(mult1) + 64'(mult2) + 64'(mult3)
                                 - 64'(mult4) - 64'(mult5);
                    state <= REDUCE;
                end
                REDUCE: begin
                    out_sample <= new_y;
                    out_channel <= held_channel;
                    state <= OUTPUT;
                end
                OUTPUT: if (out_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
