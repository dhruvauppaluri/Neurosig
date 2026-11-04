`timescale 1ns/1ps
`default_nettype none

// Formal contract for rtl/spike_count_decoder.sv.
//
// The decoder is combinational. Each of 32 channels contributes
// signed_weight_Q8.8 * unsigned_count. The score is that sum plus the
// Q8.8 bias, wrapped to 32 bits. prediction is 1 when the wrapped score
// is greater than or equal to zero, so a zero score is class 1.
//
// The reference multiplies in a 25-bit product (the exact range of a
// signed 16-bit weight times an unsigned 8-bit count) and accumulates in
// 64 bits, then keeps the low 32 bits. That matches the RTL's 32-bit
// chain of adds, including two's-complement wrap.
//
// `sva_impl(A, C) is the overlapping implication A |-> C, written as a
// boolean so the slang elaborator can compile it.

`define sva_impl(antecedent, consequent) ((!(antecedent)) || (consequent))

module spike_count_decoder_formal (
    input logic clk,
    input logic [255:0] counts,
    input logic [511:0] weights,
    input logic signed [31:0] bias
);
    logic signed [31:0] score;
    logic prediction;

    spike_count_decoder dut (
        .counts(counts),
        .weights(weights),
        .bias(bias),
        .score(score),
        .prediction(prediction)
    );

    function automatic logic signed [24:0] channel_product(
        input logic signed [15:0] weight,
        input logic [7:0] count
    );
        channel_product = 25'(weight) * 25'({1'b0, count});
    endfunction

    logic signed [63:0] spec_score;
    logic weights_are_zero;
    logic weights_are_non_negative;

    always_comb begin
        spec_score = 64'(bias);
        weights_are_zero = 1'b1;
        weights_are_non_negative = 1'b1;
        for (int ch = 0; ch < 32; ch++) begin
            spec_score += 64'(channel_product(
                weights[16*ch +: 16], counts[8*ch +: 8]));
            weights_are_zero &= (weights[16*ch +: 16] == 16'h0000);
            weights_are_non_negative &= (weights[16*ch + 15] == 1'b0);
        end
    end

    a_prediction_follows_sign: assert property (@(posedge clk)
        prediction == (score >= 32'sd0));

    a_zero_score_is_class_one: assert property (@(posedge clk)
        `sva_impl(score == 32'sd0, prediction));

    a_score_is_wrapped_dot_product: assert property (@(posedge clk)
        score == $signed(spec_score[31:0]));

    a_zero_counts_keep_bias: assert property (@(posedge clk)
        `sva_impl(counts == '0,
            score == bias && prediction == (bias >= 32'sd0)));

    a_zero_weights_keep_bias: assert property (@(posedge clk)
        `sva_impl(weights_are_zero, score == bias));

    // 32 * 32767 * 255 = 267_378_720, which is less than 2^31.
    // With a zero bias and no negative weight, the wrapped score stays
    // in the non-negative class.
    a_nonnegative_weights_zero_bias_are_class_one: assert property (@(posedge clk)
        `sva_impl(bias == 32'sd0 && weights_are_non_negative,
            score >= 32'sd0 && prediction));

    c_class_zero: cover property (@(posedge clk)
        prediction == 1'b0 && score < 32'sd0);
    c_class_one: cover property (@(posedge clk)
        prediction == 1'b1 && score > 32'sd0);
    c_tie_at_zero: cover property (@(posedge clk)
        score == 32'sd0 && prediction == 1'b1);
    c_count_moves_score: cover property (@(posedge clk)
        counts != '0 && score != bias);
endmodule

`undef sva_impl
`default_nettype wire
