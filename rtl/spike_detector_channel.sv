`timescale 1ns/1ps

module spike_detector_channel(

    input logic clk,
    input logic rst_n,

    input logic sample_valid,
    input logic end_of_stream,
    input logic signed [23:0] sample_code,
    input logic signed [23:0] threshold_code,

    output logic spike_valid,
    output logic signed [23:0] spike_code,

    input logic [31:0] sample_index,
    output logic [31:0] spike_index
);

logic signed [23:0] previous_sample;
logic have_previous;

logic downward_crossing;
assign downward_crossing = sample_valid && have_previous && (previous_sample >= threshold_code) && (sample_code < threshold_code);

logic searching;
logic signed [23:0] lowest_code;
logic [31:0] lowest_index;
logic [4:0] search_count;

logic have_last_peak;
logic [31:0] last_peak_index;

logic signed [23:0] candidate_code;
logic [31:0] candidate_index;

assign candidate_code =
    (sample_code < lowest_code) ? sample_code : lowest_code;
assign candidate_index =
    (sample_code < lowest_code) ? sample_index : lowest_index;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        previous_sample <= '0;
        have_previous <= 1'b0;
        searching <= 1'b0;
        lowest_code <= '0;
        lowest_index <= '0;
        search_count <= '0;
        spike_valid <= 1'b0;
        spike_code <= '0;
        spike_index <= '0;
        have_last_peak <= 1'b0;
        last_peak_index <= '0;


    end else begin
        spike_valid <= 1'b0;
            if (sample_valid) begin
            previous_sample <= sample_code;
            have_previous <= 1'b1;

            if (!searching && downward_crossing && (!have_last_peak || (sample_index - last_peak_index >= 32'd32))) begin
            searching <= 1'b1;
            lowest_code <= sample_code;
            lowest_index <= sample_index;
            search_count <= 5'd0;
        end else if (searching) begin

            search_count <= search_count + 5'd1;

            if (sample_code < lowest_code) begin
                lowest_code <= sample_code;
                lowest_index <= sample_index;
            end

            if (search_count == 5'd15) begin
                searching <= 1'b0;

                if(!have_last_peak || (candidate_index - last_peak_index >= 32'd32)) begin
                    spike_valid <= 1'b1;
                    spike_code <= candidate_code;
                    spike_index <= candidate_index;
                    last_peak_index <= candidate_index;
                    have_last_peak <= 1'b1;

                end
            end
        end
            end
            if (end_of_stream && searching && !sample_valid) begin
                searching <= 1'b0;
                if (!have_last_peak || (lowest_index - last_peak_index >= 32'd32)) begin
                    spike_valid <= 1'b1;
                    spike_code <= lowest_code;
                    spike_index <= lowest_index;
                    last_peak_index <= lowest_index;
                    have_last_peak <= 1'b1;
                end
            end
    end
end

endmodule
