`timescale 1ns/1ps

module sample_input_tb;
    logic clk;
    always #5 clk = ~clk;

    logic rst_n;
    logic sample_valid = 1'b0;
    logic sample_ready;
    logic signed [15:0] sample_code = '0;
    logic [4:0] channel_id = '0;
    logic captured_valid;
    logic captured_ready = 1'b0;
    logic signed [15:0] captured_code;
    logic [4:0] captured_channel;
    int checks = 0;

    sample_input #(.CHANNELS(32)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .sample_valid(sample_valid),
        .sample_ready(sample_ready),
        .sample_code(sample_code),
        .channel_id(channel_id),
        .captured_valid(captured_valid),
        .captured_ready(captured_ready),
        .captured_code(captured_code),
        .captured_channel(captured_channel)
    );

    task automatic check_output(
        input logic expected_ready,
        input logic expected_valid,
        input logic signed [15:0] expected_code,
        input logic [4:0] expected_channel
    );
        if (sample_ready !== expected_ready ||
            captured_valid !== expected_valid ||
            captured_code !== expected_code ||
            captured_channel !== expected_channel) begin
            $fatal(1, "check %0d: got ready=%b valid=%b code=%0d channel=%0d; expected %b %b %0d %0d",
                   checks, sample_ready, captured_valid, captured_code, captured_channel,
                   expected_ready, expected_valid, expected_code, expected_channel);
        end
        checks++;
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b1;
        #1;
        rst_n = 1'b0; // Falling edge triggers asynchronous reset.
        #1;
        check_output(1'b0, 1'b0, 16'sd0, 5'd0);

        // A negative sample fills the buffer while the consumer is stalled.
        @(negedge clk);
        rst_n = 1'b1;
        sample_valid = 1'b1;
        sample_code = -16'sd123;
        channel_id = 5'd7;
        #1;
        check_output(1'b1, 1'b0, 16'sd0, 5'd0);
        @(posedge clk);
        #1;
        check_output(1'b0, 1'b1, -16'sd123, 5'd7);

        // Full and not consumed: a presented new sample cannot overwrite it.
        @(negedge clk);
        sample_code = 16'sd456;
        channel_id = 5'd3;
        @(posedge clk);
        #1;
        check_output(1'b0, 1'b1, -16'sd123, 5'd7);
        @(posedge clk);
        #1;
        check_output(1'b0, 1'b1, -16'sd123, 5'd7);

        // Consume A and accept B on the same edge.
        @(negedge clk);
        captured_ready = 1'b1;
        #1;
        check_output(1'b1, 1'b1, -16'sd123, 5'd7);
        @(posedge clk);
        #1;
        check_output(1'b1, 1'b1, 16'sd456, 5'd3);

        // Consume B without a replacement: valid clears, stored data holds.
        @(negedge clk);
        sample_valid = 1'b0;
        @(posedge clk);
        #1;
        check_output(1'b1, 1'b0, 16'sd456, 5'd3);
        @(negedge clk);
        captured_ready = 1'b0;
        @(posedge clk);
        #1;
        check_output(1'b1, 1'b0, 16'sd456, 5'd3);

        // Fill with the most negative signed ADC code; stall again.
        @(negedge clk);
        sample_valid = 1'b1;
        sample_code = -16'sd32768;
        channel_id = 5'd31;
        @(posedge clk);
        #1;
        check_output(1'b0, 1'b1, -16'sd32768, 5'd31);
        @(negedge clk);
        sample_valid = 1'b0;
        @(posedge clk);
        #1;
        check_output(1'b0, 1'b1, -16'sd32768, 5'd31);

        // Reset clears a full, stalled buffer before the next edge.
        @(negedge clk);
        rst_n = 1'b0;
        #1;
        check_output(1'b0, 1'b0, 16'sd0, 5'd0);
        @(posedge clk);
        #1;
        check_output(1'b0, 1'b0, 16'sd0, 5'd0);

        // Reset release permits a new sample on the next edge.
        @(negedge clk);
        rst_n = 1'b1;
        sample_valid = 1'b1;
        sample_code = 16'sd32767;
        channel_id = 5'd0;
        @(posedge clk);
        #1;
        check_output(1'b0, 1'b1, 16'sd32767, 5'd0);

        $display("PASS sample_input: %0d checks", checks);
        $finish;
    end
endmodule
