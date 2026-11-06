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
    logic [31:0] arrival [0:DEPTH-1];
    logic [31:0] next_arrival;
    logic [DEPTH-1:0] occupied;
    logic [DEPTH-1:0] in_group;
    logic [31:0] anchor;
    integer selected, removed;
    integer free_slot;
    logic emit;
    logic signed [24:0] best_margin;
    logic [31:0] best_index;
    logic [31:0] best_arrival;
    logic signed [23:0] selected_code;
    logic [31:0] selected_index;
    logic [CHANNEL_BITS-1:0] selected_channel;

    always_comb begin
        anchor = 32'hffffffff;
        selected = -1;
        best_margin = '0;
        best_index = '0;
        best_arrival = '0;
        selected_code = '0;
        selected_index = '0;
        selected_channel = '0;
        removed = 0;
        free_slot = -1;
        in_group = '0;
        for (int i = 0; i < DEPTH; i++)
            if (occupied[i] && indices[i] < anchor) anchor = indices[i];
        for (int i = 0; i < DEPTH; i++) begin
            if (occupied[i] && indices[i] <= anchor + 32'd16) begin
                in_group[i] = 1'b1;
                removed++;
                if (selected < 0 || margins[i] > best_margin ||
                    (margins[i] == best_margin &&
                     (indices[i] < best_index ||
                      (indices[i] == best_index && arrival[i] < best_arrival)))) begin
                    selected = i;
                    best_margin = margins[i];
                    best_index = indices[i];
                    best_arrival = arrival[i];
                    selected_code = codes[i];
                    selected_index = indices[i];
                    selected_channel = channels[i];
                end
            end
        end
        emit = (count != 0) &&
               (flush || (completed_frames > anchor + 32'd32));
        for (int i = 0; i < DEPTH; i++)
            if (free_slot < 0 && (!occupied[i] || (emit && in_group[i])))
                free_slot = i;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= '0;
            occupied <= '0;
            next_arrival <= 0;
            event_valid <= 1'b0;
            event_code <= '0;
            event_index <= '0;
            event_channel <= '0;
            overflow <= 1'b0;
        end else begin
            event_valid <= emit;
            if (emit) begin
                event_code <= selected_code;
                event_index <= selected_index;
                event_channel <= selected_channel;
                for (int i = 0; i < DEPTH; i++)
                    if (in_group[i]) occupied[i] <= 1'b0;
            end
            if (candidate_valid) begin
                if (free_slot >= 0) begin
                    codes[free_slot] <= candidate_code;
                    indices[free_slot] <= candidate_index;
                    channels[free_slot] <= candidate_channel;
                    margins[free_slot] <= candidate_margin;
                    arrival[free_slot] <= next_arrival;
                    occupied[free_slot] <= 1'b1;
                    next_arrival <= next_arrival + 1'b1;
                end else overflow <= 1'b1;
            end
            count <= COUNT_BITS'(int'(count) - (emit ? removed : 0) +
                    ((candidate_valid && free_slot >= 0) ? 1 : 0));
        end
    end
endmodule
