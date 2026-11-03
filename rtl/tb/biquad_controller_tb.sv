`timescale 1ns/1ps

module biquad_controller_tb;
    localparam int CHANNELS = 4;
    localparam int CHANNEL_BITS = $clog2(CHANNELS);
    localparam logic signed [31:0] ONE = 32'sd1073741824;
    localparam logic signed [31:0] HALF = 32'sd536870912;
    localparam logic signed [31:0] QUARTER = 32'sd268435456;

    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 1;
    logic in_valid = 0;
    logic in_ready;
    logic signed [23:0] in_sample = 0;
    logic [CHANNEL_BITS-1:0] in_channel = 0;
    logic out_valid;
    logic out_ready = 0;
    logic signed [23:0] out_sample;
    logic [CHANNEL_BITS-1:0] out_channel;
    int checks = 0;

    biquad_controller #(.CHANNELS(CHANNELS)) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_ready(in_ready),
        .in_sample(in_sample), .in_channel(in_channel),
        .out_valid(out_valid), .out_ready(out_ready),
        .out_sample(out_sample), .out_channel(out_channel),
        .b0(ONE), .b1(HALF), .b2(QUARTER),
        .a1(32'sd0), .a2(32'sd0)
    );

    task automatic check(input logic condition, input string message);
        if (condition !== 1'b1) $fatal(1, "FAIL: %s", message);
        checks++;
    endtask

    task automatic send_sample(input logic signed [23:0] value,
                               input logic [CHANNEL_BITS-1:0] channel);
        @(negedge clk);
        check(in_ready && !out_valid, "ready before input");
        in_valid = 1;
        in_sample = value;
        in_channel = channel;
        @(posedge clk);
        #1;
        check(!in_ready && !out_valid, "COMPUTE after acceptance");
        @(negedge clk);
        in_valid = 0;
    endtask

    task automatic expect_output(input logic signed [23:0] value,
                                 input logic [CHANNEL_BITS-1:0] channel);
        @(posedge clk);
        #1;
        check(out_valid && !in_ready, "OUTPUT after compute");
        check(out_sample === value, "filtered result");
        check(out_channel === channel, "output channel");
    endtask

    task automatic consume_output;
        @(negedge clk);
        out_ready = 1;
        @(posedge clk);
        #1;
        check(!out_valid && in_ready, "IDLE after output handshake");
        @(negedge clk);
        out_ready = 0;
    endtask

    initial begin
        #1;
        rst_n = 0;
        #1;
        check(!out_valid, "invalid during reset");
        @(negedge clk);
        rst_n = 1;
        #1;
        check(in_ready, "ready after reset");

        send_sample(24'sd100, 0);
        expect_output(24'sd100, 0);
        // Keep a new input valid while the output is stalled: it must not be accepted.
        @(negedge clk);
        in_valid = 1;
        in_sample = 24'sd999;
        in_channel = 2;
        repeat (3) begin
            @(posedge clk);
            #1;
            check(out_valid && !in_ready, "backpressure holds OUTPUT");
            check(out_sample === 24'sd100 && out_channel === 0,
                  "stalled output stays stable");
        end
        @(negedge clk);
        in_valid = 0;
        consume_output();

        send_sample(-24'sd40, 1);
        expect_output(-24'sd40, 1);
        consume_output();
        send_sample(24'sd200, 0);
        expect_output(24'sd250, 0); // 200 + 100/2
        consume_output();
        send_sample(24'sd300, 0);
        expect_output(24'sd425, 0); // 300 + 200/2 + 100/4
        consume_output();
        send_sample(24'sd20, 1);
        expect_output(24'sd0, 1); // 20 - 40/2; channel 0 history is isolated
        consume_output();
        send_sample(24'sd10, 2);
        expect_output(24'sd10, 2); // stalled channel-2 input was discarded

        // Reset even while holding a result: output and all channel histories clear.
        @(negedge clk);
        rst_n = 0;
        #1;
        check(!out_valid && !out_sample && !out_channel, "reset clears output");
        @(negedge clk);
        rst_n = 1;
        send_sample(24'sd10, 0);
        expect_output(24'sd10, 0);
        $display("PASS: biquad_controller_tb (%0d checks)", checks);
        $finish;
    end
endmodule
