`timescale 1ns/1ps

// ADC code stream -> two-stage IIR -> per-channel candidates -> event merge.
// Supply threshold_codes from offline calibration and assert flush after the
// final filtered sample has cleared the pipeline.
module neurosig_pipeline #(
    parameter int CHANNELS = 32,
    parameter int CHANNEL_BITS = (CHANNELS > 1 ? $clog2(CHANNELS) : 1)
) (
    input  logic clk,
    input  logic rst_n,
    input  logic sample_valid,
    output logic sample_ready,
    input  logic signed [15:0] sample_code,
    input  logic [CHANNEL_BITS-1:0] channel_id,
    input  logic [CHANNELS*24-1:0] threshold_codes,
    input  logic flush,
    output logic event_valid,
    output logic signed [23:0] event_code,
    output logic [31:0] event_index,
    output logic [CHANNEL_BITS-1:0] event_channel,
    output logic event_overflow
);
    logic captured_valid, captured_ready;
    logic signed [15:0] captured_code;
    logic [CHANNEL_BITS-1:0] captured_channel;
    logic filtered_valid, filtered_ready;
    logic signed [23:0] filtered_code;
    logic [CHANNEL_BITS-1:0] filtered_channel;
    logic candidate_valid;
    logic signed [23:0] candidate_code;
    logic [31:0] candidate_index, completed_frames;
    logic [CHANNEL_BITS-1:0] candidate_channel;
    logic signed [24:0] candidate_margin;
    logic [CHANNEL_BITS:0] end_count;
    logic channel_end;
    logic merge_flush;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) end_count <= '0;
        else if (flush && int'(end_count) < CHANNELS + 2)
            end_count <= end_count + 1'b1;
    end
    assign channel_end = flush && int'(end_count) < CHANNELS;
    assign merge_flush = flush && int'(end_count) >= CHANNELS + 1;

    sample_input #(.CHANNELS(CHANNELS), .CHANNEL_BITS(CHANNEL_BITS)) input_buffer (
        .clk(clk), .rst_n(rst_n),
        .sample_valid(sample_valid), .sample_ready(sample_ready),
        .sample_code(sample_code), .channel_id(channel_id),
        .captured_valid(captured_valid), .captured_ready(captured_ready),
        .captured_code(captured_code), .captured_channel(captured_channel)
    );

    spike_band_filter #(.CHANNELS(CHANNELS), .CHANNEL_BITS(CHANNEL_BITS)) filter (
        .clk(clk), .rst_n(rst_n),
        .in_valid(captured_valid), .in_ready(captured_ready),
        .in_sample({{8{captured_code[15]}}, captured_code}),
        .in_channel(captured_channel),
        .out_valid(filtered_valid), .out_ready(filtered_ready),
        .out_sample(filtered_code), .out_channel(filtered_channel)
    );

    spike_detector_multichannel #(.CHANNELS(CHANNELS), .CHANNEL_BITS(CHANNEL_BITS)) detector (
        .clk(clk), .rst_n(rst_n),
        .in_valid(filtered_valid), .in_ready(filtered_ready),
        .in_sample(filtered_code), .in_channel(filtered_channel),
        .threshold_codes(threshold_codes),
        .end_of_stream(channel_end), .end_channel(CHANNEL_BITS'(end_count)),
        .event_valid(candidate_valid), .event_code(candidate_code),
        .event_index(candidate_index), .event_channel(candidate_channel),
        .event_margin(candidate_margin), .completed_frames(completed_frames)
    );

    spike_event_merge #(.CHANNEL_BITS(CHANNEL_BITS)) merger (
        .clk(clk), .rst_n(rst_n), .flush(merge_flush),
        .completed_frames(completed_frames),
        .candidate_valid(candidate_valid), .candidate_code(candidate_code),
        .candidate_index(candidate_index), .candidate_channel(candidate_channel),
        .candidate_margin(candidate_margin),
        .event_valid(event_valid), .event_code(event_code),
        .event_index(event_index), .event_channel(event_channel),
        .overflow(event_overflow)
    );
endmodule
