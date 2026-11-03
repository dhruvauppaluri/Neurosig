`timescale 1ns/1ps

// Merge candidates within 16 sample frames of the earliest candidate,
// retaining the largest threshold margin (earliest index wins ties).
// completed_frames is the minimum accepted sample count over all channels.
module spike_event_merge #(
    parameter int CHANNEL_BITS = 5,
    parameter int DEPTH = 64
) (
    input  logic clk,
    input  logic rst_n,
    input  logic flush,
    input  logic [31:0] completed_frames,
    input  logic candidate_valid,
    input  logic signed [23:0] candidate_code,
    input  logic [31:0] candidate_index,
    input  logic [CHANNEL_BITS-1:0] candidate_channel,
    input  logic signed [24:0] candidate_margin,
    output logic event_valid,
    output logic signed [23:0] event_code,
    output logic [31:0] event_index,
    output logic [CHANNEL_BITS-1:0] event_channel,
    output logic overflow
);
    localparam int COUNT_BITS = $clog2(DEPTH + 1);
    logic [COUNT_BITS-1:0] count;
    logic signed [23:0] codes [0:DEPTH-1];
    logic [31:0] indices [0:DEPTH-1];
    logic [CHANNEL_BITS-1:0] channels [0:DEPTH-1];
    logic signed [24:0] margins [0:DEPTH-1];
    logic [DEPTH-1:0] in_group;
    logic [31:0] anchor;
    integer selected, remaining;
    integer append_at;
    logic emit;

    always_comb begin
        anchor = 32'hffffffff;
        selected = -1;
        remaining = 0;
        in_group = '0;
        for (int i = 0; i < DEPTH; i++)
            if (i < count && indices[i] < anchor) anchor = indices[i];
        for (int i = 0; i < DEPTH; i++) begin
            if (i < count && indices[i] <= anchor + 32'd16) begin
                in_group[i] = 1'b1;
                if (selected < 0 || margins[i] > margins[selected] ||
                    (margins[i] == margins[selected] && indices[i] < indices[selected]))
                    selected = i;
            end else if (i < count) remaining++;
        end
        emit = (count != 0) &&
               (flush || (completed_frames > anchor + 32'd32));
        append_at = emit ? remaining : int'(count);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= '0;
            event_valid <= 1'b0;
            event_code <= '0;
            event_index <= '0;
            event_channel <= '0;
            overflow <= 1'b0;
        end else begin
            event_valid <= emit;
            if (emit) begin
                event_code <= codes[selected];
                event_index <= indices[selected];
                event_channel <= channels[selected];
                // Preserve arrival order among candidates outside this group.
                begin : compact
                    integer dest;
                    dest = 0;
                    for (int i = 0; i < DEPTH; i++) begin
                        if (i < count && !in_group[i]) begin
                            codes[dest] <= codes[i];
                            indices[dest] <= indices[i];
                            channels[dest] <= channels[i];
                            margins[dest] <= margins[i];
                            dest++;
                        end
                    end
                end
            end
            if (candidate_valid) begin
                if (append_at < DEPTH) begin
                    codes[append_at] <= candidate_code;
                    indices[append_at] <= candidate_index;
                    channels[append_at] <= candidate_channel;
                    margins[append_at] <= candidate_margin;
                end else overflow <= 1'b1;
            end
            count <= COUNT_BITS'(append_at +
                    ((candidate_valid && append_at < DEPTH) ? 1 : 0));
        end
    end
endmodule
