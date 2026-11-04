`timescale 1ns/1ps
`default_nettype none

// Formal contract for rtl/biquad_controller.sv and the per-channel delay
// line in rtl/biquad_state.sv.
//
// One accepted sample takes two clocks to become a valid output:
//   IDLE --accept--> COMPUTE --always--> OUTPUT --out_ready--> IDLE
// OUTPUT holds out_sample and out_channel while out_ready is low, and
// in_ready stays low for that whole interval. Channel memories shift only
// for the accepted channel, on the COMPUTE clock. new_y is whatever the
// arithmetic unit drives during COMPUTE; this proof checks that the
// controller registers that value. formal/biquad_arithmetic_blackbox.sv
// replaces the arithmetic unit in this proof, so new_y is a free input.
// Integer arithmetic stays with rtl/tb/biquad_arithmetic_tb.sv.
//
// Encoding locked below matches the enum order in biquad_controller.sv:
// IDLE = 0, COMPUTE = 1, OUTPUT = 2. The unused 2'd3 encoding steps to IDLE.
//
// in_ready is (state == IDLE), including while rst_n is low. The reset
// branch does not capture in_sample, so a ready/valid handshake during
// reset is not an accept. sample_input qualifies its own ready with rst_n.
//
// `sva_impl(A, C) is the overlapping implication A |-> C, written as a
// boolean so the slang elaborator can compile it.

`define sva_impl(antecedent, consequent) ((!(antecedent)) || (consequent))

module biquad_controller_formal #(
    parameter int CHANNELS = 32,
    parameter int CHANNEL_BITS = (CHANNELS > 1 ? $clog2(CHANNELS) : 1)
) (
    input logic clk,
    input logic rst_n,
    input logic in_valid,
    input logic signed [23:0] in_sample,
    input logic [CHANNEL_BITS-1:0] in_channel,
    input logic out_ready,
    input logic signed [31:0] b0,
    input logic signed [31:0] b1,
    input logic signed [31:0] b2,
    input logic signed [31:0] a1,
    input logic signed [31:0] a2
);
    localparam logic [1:0] F_IDLE = 2'd0;
    localparam logic [1:0] F_COMPUTE = 2'd1;
    localparam logic [1:0] F_OUTPUT = 2'd2;
    localparam logic [1:0] F_ILLEGAL = 2'd3;

    logic in_ready;
    logic out_valid;
    logic signed [23:0] out_sample;
    logic [CHANNEL_BITS-1:0] out_channel;

    biquad_controller #(
        .CHANNELS(CHANNELS),
        .CHANNEL_BITS(CHANNEL_BITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_sample(in_sample),
        .in_channel(in_channel),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_sample(out_sample),
        .out_channel(out_channel),
        .b0(b0),
        .b1(b1),
        .b2(b2),
        .a1(a1),
        .a2(a2)
    );

    // Base case starts in reset. past_ok arms $past on later cycles.
    logic past_ok = 1'b0;
    always_ff @(posedge clk)
        past_ok <= 1'b1;

    assume property (@(posedge clk) past_ok || !rst_n);

    a_idle_encoding: assert property (@(posedge clk)
        in_ready == (dut.state == F_IDLE));

    a_output_encoding: assert property (@(posedge clk)
        out_valid == (dut.state == F_OUTPUT));

    a_ready_valid_mutex: assert property (@(posedge clk)
        !(in_ready && out_valid));

    // A clock with rst_n low has completed its reset assignment by the
    // next cycle.
    a_reset_clears_controller: assert property (@(posedge clk)
        `sva_impl(past_ok && $past(!rst_n),
            dut.state == F_IDLE &&
            dut.held_sample == '0 &&
            dut.held_channel == '0 &&
            out_sample == '0 &&
            out_channel == '0 &&
            !out_valid));

    a_idle_holds_without_valid: assert property (@(posedge clk)
        `sva_impl(past_ok && rst_n && $past(rst_n) && $past(dut.state == F_IDLE) && !$past(in_valid),
            dut.state == F_IDLE));

    a_accept_enters_compute: assert property (@(posedge clk)
        `sva_impl(past_ok && rst_n && $past(rst_n) && $past(dut.state == F_IDLE) && $past(in_valid),
            dut.state == F_COMPUTE &&
            dut.held_sample == $past(in_sample) &&
            dut.held_channel == $past(in_channel)));

    a_compute_registers_result: assert property (@(posedge clk)
        `sva_impl(past_ok && rst_n && $past(rst_n) && $past(dut.state == F_COMPUTE),
            dut.state == F_OUTPUT &&
            out_valid &&
            !in_ready &&
            out_sample == $past(dut.new_y) &&
            out_channel == $past(dut.held_channel)));

    a_output_holds_under_backpressure: assert property (@(posedge clk)
        `sva_impl(past_ok && rst_n && $past(rst_n) && $past(dut.state == F_OUTPUT) && !$past(out_ready),
            dut.state == F_OUTPUT &&
            out_sample == $past(out_sample) &&
            out_channel == $past(out_channel)));

    a_output_handshake_returns_idle: assert property (@(posedge clk)
        `sva_impl(past_ok && rst_n && $past(rst_n) && $past(dut.state == F_OUTPUT) && $past(out_ready),
            dut.state == F_IDLE &&
            in_ready &&
            !out_valid));

    a_held_stable_without_accept: assert property (@(posedge clk)
        `sva_impl(past_ok && rst_n && $past(rst_n) && !$past(dut.state == F_IDLE && in_valid),
            dut.held_sample == $past(dut.held_sample) &&
            dut.held_channel == $past(dut.held_channel)));

    a_backpressure_does_not_capture: assert property (@(posedge clk)
        `sva_impl(past_ok && rst_n && $past(rst_n) && !$past(in_ready),
            dut.held_sample == $past(dut.held_sample) &&
            dut.held_channel == $past(dut.held_channel)));

    a_output_updates_only_from_compute: assert property (@(posedge clk)
        `sva_impl(past_ok && rst_n && $past(rst_n) && $past(dut.state != F_COMPUTE),
            out_sample == $past(out_sample) &&
            out_channel == $past(out_channel)));

    a_output_has_a_source: assert property (@(posedge clk)
        `sva_impl(past_ok && rst_n && $past(rst_n) && dut.state == F_OUTPUT,
            $past(dut.state == F_COMPUTE) || $past(dut.state == F_OUTPUT)));

    a_illegal_encoding_steps_to_idle: assert property (@(posedge clk)
        `sva_impl(past_ok && rst_n && $past(rst_n) && $past(dut.state == F_ILLEGAL),
            dut.state == F_IDLE));

    a_legal_encoding_stays_legal: assert property (@(posedge clk)
        `sva_impl(past_ok && ($past(!rst_n) || $past(dut.state != F_ILLEGAL)),
            dut.state != F_ILLEGAL));

    // Accept at T-2, busy at T-1 and T. With out_ready held high the next
    // accept is three cycles after the previous one.
    a_busy_for_two_cycles_after_accept: assert property (@(posedge clk)
        `sva_impl(past_ok && rst_n && $past(past_ok, 2) && $past(rst_n) && $past(rst_n, 2) &&
                 $past(in_valid && in_ready, 2),
            $past(!in_ready) && !in_ready));

    a_accepted_sample_is_presented: assert property (@(posedge clk)
        `sva_impl(past_ok && rst_n && $past(past_ok, 2) && $past(rst_n) && $past(rst_n, 2) &&
                 $past(in_valid && in_ready, 2),
            out_valid &&
            !in_ready &&
            out_channel == $past(in_channel, 2) &&
            out_sample == $past(dut.new_y) &&
            dut.held_channel == $past(in_channel, 2) &&
            dut.held_sample == $past(in_sample, 2)));

    generate
        for (genvar ch = 0; ch < CHANNELS; ch++) begin : channel
            a_reset_clears_memory: assert property (@(posedge clk)
                `sva_impl(past_ok && $past(!rst_n),
                    dut.state_storage.x1_mem[ch] == '0 &&
                    dut.state_storage.x2_mem[ch] == '0 &&
                    dut.state_storage.y1_mem[ch] == '0 &&
                    dut.state_storage.y2_mem[ch] == '0));

            a_channel_shifts_on_compute: assert property (@(posedge clk)
                `sva_impl(past_ok && rst_n && $past(rst_n) && $past(dut.state == F_COMPUTE) &&
                         $past(dut.held_channel == CHANNEL_BITS'(ch)),
                    dut.state_storage.x1_mem[ch] == $past(dut.held_sample) &&
                    dut.state_storage.x2_mem[ch] == $past(dut.state_storage.x1_mem[ch]) &&
                    dut.state_storage.y1_mem[ch] == $past(dut.new_y) &&
                    dut.state_storage.y2_mem[ch] == $past(dut.state_storage.y1_mem[ch])));

            a_other_activity_leaves_channel: assert property (@(posedge clk)
                `sva_impl(past_ok && rst_n && $past(rst_n) &&
                         !($past(dut.state == F_COMPUTE) &&
                           $past(dut.held_channel == CHANNEL_BITS'(ch))),
                    dut.state_storage.x1_mem[ch] == $past(dut.state_storage.x1_mem[ch]) &&
                    dut.state_storage.x2_mem[ch] == $past(dut.state_storage.x2_mem[ch]) &&
                    dut.state_storage.y1_mem[ch] == $past(dut.state_storage.y1_mem[ch]) &&
                    dut.state_storage.y2_mem[ch] == $past(dut.state_storage.y2_mem[ch])));
        end
    endgenerate

    c_accept_channel: cover property (@(posedge clk)
        out_valid && out_channel == CHANNEL_BITS'(3) &&
        $past(in_valid && in_ready && in_channel == CHANNEL_BITS'(3), 2));

    c_backpressure: cover property (@(posedge clk)
        out_valid && !out_ready &&
        $past(out_valid && !out_ready) &&
        $past(out_valid && !out_ready, 2));

    c_channel_switch: cover property (@(posedge clk)
        out_valid && out_channel == CHANNEL_BITS'(1) &&
        $past(in_valid && in_channel == CHANNEL_BITS'(1), 2) &&
        $past(out_valid && out_channel == CHANNEL_BITS'(0) && out_ready, 3));

    c_ready_high_during_reset: cover property (@(posedge clk)
        !rst_n && in_ready && !out_valid);

    // new_y has to be free: a nonzero code is presented two cycles after accept.
    c_nonzero_sample: cover property (@(posedge clk)
        out_valid && out_sample == 24'sd1234 &&
        out_channel == CHANNEL_BITS'(3) &&
        $past(in_valid && in_ready && in_channel == CHANNEL_BITS'(3), 2));
endmodule

`undef sva_impl
`default_nettype wire
