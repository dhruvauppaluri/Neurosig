`timescale 1ns/1ps

module spike_band_filter_tb;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 1;
    logic in_valid = 0, in_ready;
    logic signed [23:0] in_sample = 0;
    logic [1:0] in_channel = 0;
    logic out_valid, out_ready = 0;
    logic signed [23:0] out_sample;
    logic [1:0] out_channel;

    spike_band_filter #(.CHANNELS(4)) dut (.*);

    task automatic send(input logic [1:0] channel);
        @(negedge clk);
        while (!in_ready) @(negedge clk);
        in_sample = 24'sd1000;
        in_channel = channel;
        in_valid = 1;
        @(posedge clk);
        #1;
        if (in_ready) $fatal(1, "input was not captured");
        @(negedge clk);
        in_valid = 0;
    endtask

    task automatic expect_first(input logic [1:0] channel);
        @(negedge clk);
        while (!out_valid) @(negedge clk);
        if (out_sample !== 24'sd179 || out_channel !== channel)
            $fatal(1, "wrong output: sample=%0d channel=%0d", out_sample, out_channel);
        repeat (3) begin
            @(posedge clk);
            #1;
            if (!out_valid || out_sample !== 24'sd179 || out_channel !== channel)
                $fatal(1, "output changed during stall");
        end
        @(negedge clk);
        out_ready = 1;
        @(posedge clk);
        #1;
        if (out_valid) $fatal(1, "output not consumed");
        @(negedge clk);
        out_ready = 0;
    endtask

    initial begin
        #1 rst_n = 0;
        @(negedge clk);
        rst_n = 1;
        send(0);
        expect_first(0);
        send(1); // separate channel state must produce the same first-sample output
        expect_first(1);
        $display("PASS: spike_band_filter_tb (two channels, handshakes, stalls)");
        $finish;
    end
endmodule
