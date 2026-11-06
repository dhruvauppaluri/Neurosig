`timescale 1ns/1ps

// Bounded, one-slot-per-clock event merge for FPGA implementation. Candidate
// input is never stalled; a full queue raises sticky overflow. Selection uses
// the same anchor, margin, sample-index, and arrival-order rules as the
// combinational reference merger.
module spike_event_merge_serial #(
    parameter int CHANNEL_BITS = 5,
    parameter int DEPTH = 32
) (
    input logic clk,
    input logic rst_n,
    input logic flush,
    input logic [31:0] completed_frames,
    input logic candidate_valid,
    input logic signed [23:0] candidate_code,
    input logic [31:0] candidate_index,
    input logic [CHANNEL_BITS-1:0] candidate_channel,
    input logic signed [24:0] candidate_margin,
    output logic event_valid,
    output logic signed [23:0] event_code,
    output logic [31:0] event_index,
    output logic [CHANNEL_BITS-1:0] event_channel,
    output logic overflow
);
    localparam int SLOT_BITS = $clog2(DEPTH);
    typedef enum logic [1:0] {FIND, WAIT, SELECT} state_t;
    state_t state;
    logic [DEPTH-1:0] occupied, group_mask;
    logic signed [23:0] codes [0:DEPTH-1];
    logic [31:0] indices [0:DEPTH-1];
    logic [CHANNEL_BITS-1:0] channels [0:DEPTH-1];
    logic signed [24:0] margins [0:DEPTH-1];
    logic [31:0] arrivals [0:DEPTH-1];
    logic [31:0] next_arrival, scan_min, anchor;
    logic [SLOT_BITS-1:0] scan_slot;
    logic best_valid;
    logic signed [24:0] best_margin;
    logic [31:0] best_index, best_arrival;
    logic signed [23:0] best_code;
    logic [CHANNEL_BITS-1:0] best_channel;
    logic [2:0] quiet_count;
    logic free_found;
    logic [SLOT_BITS-1:0] free_slot;
    logic [31:0] min_next;
    logic member, better, restart_select;

    always_comb begin
        free_found = 1'b0;
        free_slot = '0;
        for (int i=0; i<DEPTH; i++) begin
            if (!free_found && !occupied[i]) begin
                free_found = 1'b1;
                free_slot = SLOT_BITS'(i);
            end
        end
        min_next = scan_min;
        if (occupied[scan_slot] && indices[scan_slot] < min_next)
            min_next = indices[scan_slot];
        if (candidate_valid && free_found && candidate_index < min_next)
            min_next = candidate_index;
        member = occupied[scan_slot] && indices[scan_slot] <= anchor + 32'd16;
        better = member && (!best_valid || margins[scan_slot] > best_margin ||
                 (margins[scan_slot] == best_margin &&
                  (indices[scan_slot] < best_index ||
                   (indices[scan_slot] == best_index &&
                    arrivals[scan_slot] < best_arrival))));
        restart_select = candidate_valid && free_found &&
                         candidate_index <= anchor + 32'd16;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= FIND;
            occupied <= '0;
            group_mask <= '0;
            scan_slot <= '0;
            scan_min <= 32'hffffffff;
            anchor <= 32'hffffffff;
            next_arrival <= 0;
            best_valid <= 0;
            best_margin <= 0;
            best_index <= 0;
            best_arrival <= 0;
            best_code <= 0;
            best_channel <= 0;
            quiet_count <= 0;
            event_valid <= 0;
            event_code <= 0;
            event_index <= 0;
            event_channel <= 0;
            overflow <= 0;
        end else begin
            event_valid <= 1'b0;
            if (candidate_valid) begin
                quiet_count <= 0;
                if (free_found) begin
                    codes[free_slot] <= candidate_code;
                    indices[free_slot] <= candidate_index;
                    channels[free_slot] <= candidate_channel;
                    margins[free_slot] <= candidate_margin;
                    arrivals[free_slot] <= next_arrival;
                    occupied[free_slot] <= 1'b1;
                    next_arrival <= next_arrival + 1'b1;
                end else overflow <= 1'b1;
            end else if (quiet_count != 3'd7) quiet_count <= quiet_count + 1'b1;

            case (state)
                FIND: begin
                    scan_min <= min_next;
                    if (scan_slot == SLOT_BITS'(DEPTH-1)) begin
                        anchor <= min_next;
                        scan_slot <= '0;
                        state <= WAIT;
                    end else scan_slot <= scan_slot + 1'b1;
                end
                WAIT: begin
                    if (candidate_valid && free_found && candidate_index < anchor)
                        anchor <= candidate_index;
                    if (anchor != 32'hffffffff &&
                        !(candidate_valid && free_found && candidate_index < anchor) &&
                        ((completed_frames > anchor + 32'd32) ||
                         (flush && quiet_count >= 3'd4))) begin
                        scan_slot <= '0;
                        group_mask <= '0;
                        best_valid <= 1'b0;
                        state <= SELECT;
                    end else if (anchor == 32'hffffffff && candidate_valid && free_found)
                        anchor <= candidate_index;
                end
                SELECT: begin
                    if (restart_select) begin
                        scan_slot <= '0;
                        group_mask <= '0;
                        best_valid <= 1'b0;
                    end else begin
                        if (member) group_mask[scan_slot] <= 1'b1;
                        if (better) begin
                            best_valid <= 1'b1;
                            best_margin <= margins[scan_slot];
                            best_index <= indices[scan_slot];
                            best_arrival <= arrivals[scan_slot];
                            best_code <= codes[scan_slot];
                            best_channel <= channels[scan_slot];
                        end
                        if (scan_slot == SLOT_BITS'(DEPTH-1)) begin
                            if (best_valid || member) begin
                                event_valid <= 1'b1;
                                event_code <= better ? codes[scan_slot] : best_code;
                                event_index <= better ? indices[scan_slot] : best_index;
                                event_channel <= better ? channels[scan_slot] : best_channel;
                                occupied <= (occupied & ~(group_mask |
                                             (member ? ({{(DEPTH-1){1'b0}},1'b1} << scan_slot) : '0))) |
                                            ((candidate_valid && free_found) ?
                                             ({{(DEPTH-1){1'b0}},1'b1} << free_slot) : '0);
                            end
                            scan_slot <= '0;
                            scan_min <= 32'hffffffff;
                            state <= FIND;
                        end else scan_slot <= scan_slot + 1'b1;
                    end
                end
                default: state <= FIND;
            endcase
        end
    end
endmodule
