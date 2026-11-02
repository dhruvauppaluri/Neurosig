`timescale 1ns/1ps

module biquad_state_tb;
    logic clk;
    always #5 clk = ~clk;

    logic rst_n;
    logic [4:0] read_channel = '0;
    logic signed [23:0] x1, x2, y1, y2;
    logic write_enable = 1'b0;
    logic [4:0] write_channel = '0;
    logic signed [23:0] new_x = '0;
    logic signed [23:0] new_y = '0;
    int checks = 0;

    biquad_state #(.CHANNELS(32)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .read_channel(read_channel),
        .x1(x1), .x2(x2), .y1(y1), .y2(y2),
        .write_enable(write_enable),
        .write_channel(write_channel),
        .new_x(new_x), .new_y(new_y)
    );

    task automatic expect_state(
        input logic [4:0] channel,
        input logic signed [23:0] expected_x1,
        input logic signed [23:0] expected_x2,
        input logic signed [23:0] expected_y1,
        input logic signed [23:0] expected_y2
    );
        read_channel = channel;
        #1; // Allow the combinational read outputs to update.
        if (x1 !== expected_x1 || x2 !== expected_x2 ||
            y1 !== expected_y1 || y2 !== expected_y2) begin
            $fatal(1, "check %0d channel %0d: got (%0d,%0d,%0d,%0d), expected (%0d,%0d,%0d,%0d)",
                checks, channel, x1, x2, y1, y2,
                expected_x1, expected_x2, expected_y1, expected_y2);
        end
        checks++;
    endtask

    task automatic write_state(
        input logic [4:0] channel,
        input logic signed [23:0] input_value,
        input logic signed [23:0] output_value
    );
        @(negedge clk);
        write_enable = 1'b1;
        write_channel = channel;
        new_x = input_value;
        new_y = output_value;
        @(posedge clk);
        #1; // Let nonblocking memory writes settle.
        write_enable = 1'b0;
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b1;
        #1;
        rst_n = 1'b0;
        #1;
        expect_state(5'd7, 0, 0, 0, 0);
        expect_state(5'd8, 0, 0, 0, 0);
        expect_state(5'd31, 0, 0, 0, 0);

        @(negedge clk);
        rst_n = 1'b1;
        write_state(5'd7, 24'sd20, -24'sd30);
        expect_state(5'd7, 24'sd20, 0, -24'sd30, 0);
        expect_state(5'd8, 0, 0, 0, 0);

        write_state(5'd7, 24'sd40, -24'sd50);
        expect_state(5'd7, 24'sd40, 24'sd20, -24'sd50, -24'sd30);

        // Read channel 8 while writing channel 7: no cross-channel copy.
        read_channel = 5'd8;
        write_state(5'd7, -24'sd60, 24'sd70);
        expect_state(5'd8, 0, 0, 0, 0);
        expect_state(5'd7, -24'sd60, 24'sd40, 24'sd70, -24'sd50);

        write_state(5'd31, -24'sd8388608, 24'sd8388607);
        expect_state(5'd31, -24'sd8388608, 0, 24'sd8388607, 0);
        expect_state(5'd7, -24'sd60, 24'sd40, 24'sd70, -24'sd50);

        // Disabled writes leave every channel unchanged.
        @(negedge clk);
        write_channel = 5'd7;
        new_x = 24'sd999;
        new_y = 24'sd999;
        @(posedge clk);
        #1;
        expect_state(5'd7, -24'sd60, 24'sd40, 24'sd70, -24'sd50);

        // Asynchronous reset clears histories even between clock edges.
        @(negedge clk);
        rst_n = 1'b0;
        #1;
        expect_state(5'd7, 0, 0, 0, 0);
        expect_state(5'd31, 0, 0, 0, 0);

        $display("PASS biquad_state: %0d checks", checks);
        $finish;
    end
endmodule
