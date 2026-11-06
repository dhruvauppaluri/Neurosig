`timescale 1ns/1ps

// One accepted filtered sample per clock. Thresholds are offline-calibrated
// signed 24-bit codes, packed with channel 0 in the least significant slice.
module spike_detector_multichannel #(
    parameter int CHANNELS = 32,
    parameter int CHANNEL_BITS = (CHANNELS > 1 ? $clog2(CHANNELS) : 1),
    parameter bit FRAME_ORDERED = 0
) (
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    output logic in_ready,
    input  logic signed [23:0] in_sample,
    input  logic [CHANNEL_BITS-1:0] in_channel,
    input  logic [CHANNELS*24-1:0] threshold_codes,
    input  logic end_of_stream,
    input  logic [CHANNEL_BITS-1:0] end_channel,
    output logic event_valid,
    output logic signed [23:0] event_code,
    output logic [31:0] event_index,
    output logic [CHANNEL_BITS-1:0] event_channel,
    output logic signed [24:0] event_margin,
    output logic [31:0] completed_frames
);
    logic [31:0] sample_count [0:CHANNELS-1];
    logic [CHANNELS-1:0] channel_event_valid;
    logic signed [23:0] channel_event_code [0:CHANNELS-1];
    logic [31:0] channel_event_index [0:CHANNELS-1];
    logic signed [23:0] threshold [0:CHANNELS-1];
    logic [CHANNELS-1:0] accepted;
    logic invalid_channel;

    assign invalid_channel = int'(in_channel) >= CHANNELS;
    assign in_ready = rst_n && !invalid_channel;

    genvar c;
    generate for (c = 0; c < CHANNELS; c = c + 1) begin : channels
        assign threshold[c] = $signed(threshold_codes[c*24 +: 24]);
        assign accepted[c] = in_valid && in_ready && (in_channel == CHANNEL_BITS'(c));
        spike_detector_channel detector (
            .clk(clk), .rst_n(rst_n),
            .sample_valid(accepted[c]),
            .end_of_stream(end_of_stream && end_channel == CHANNEL_BITS'(c)),
            .sample_code(in_sample),
            .threshold_code(threshold[c]),
            .sample_index(sample_count[c]),
            .spike_valid(channel_event_valid[c]),
            .spike_code(channel_event_code[c]),
            .spike_index(channel_event_index[c])
        );
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) sample_count[c] <= '0;
            else if (accepted[c]) sample_count[c] <= sample_count[c] + 32'd1;
        end
    end endgenerate

    // Only one channel receives a sample per cycle, so at most one detector
    // can finish its 16-sample search on that edge.
    always_comb begin
        event_valid = 1'b0;
        event_code = '0;
        event_index = '0;
        event_channel = '0;
        event_margin = '0;
        for (int c = 0; c < CHANNELS; c++) begin
            if (channel_event_valid[c]) begin
                event_valid = 1'b1;
                event_code = channel_event_code[c];
                event_index = channel_event_index[c];
                event_channel = CHANNEL_BITS'(c);
                event_margin = $signed({threshold[c][23], threshold[c]})
                             - $signed({channel_event_code[c][23], channel_event_code[c]});
            end
        end
    end

    // The C5G mailbox validates strict frame-major order before feeding the
    // pipeline. In that mode the final channel count is the completed-frame
    // count, avoiding a long 32-channel minimum path. The generic interface
    // keeps the minimum for arbitrary channel order.
    generate if (FRAME_ORDERED) begin : ordered_frames
        assign completed_frames = sample_count[CHANNELS-1];
    end else begin : arbitrary_order
        always_comb begin
            completed_frames = sample_count[0];
            for (int c = 1; c < CHANNELS; c++)
                if (sample_count[c] < completed_frames)
                    completed_frames = sample_count[c];
        end
    end endgenerate
endmodule
