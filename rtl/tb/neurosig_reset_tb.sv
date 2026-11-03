`timescale 1ns/1ps

module neurosig_reset_tb;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 1, flush = 0;
    logic sample_valid = 0, sample_ready;
    logic signed [15:0] sample_code = 0;
    logic [1:0] channel_id = 0;
    logic [95:0] threshold_codes = '0;
    logic event_valid, event_overflow;
    logic signed [23:0] event_code;
    logic [31:0] event_index;
    logic [1:0] event_channel;

    neurosig_pipeline #(.CHANNELS(4)) dut (.*);

    task automatic send(input int value, channel);
        @(negedge clk);
        while (!sample_ready) @(negedge clk);
        sample_valid = 1;
        sample_code = 16'(value);
        channel_id = 2'(channel);
        @(posedge clk);
        @(negedge clk);
        sample_valid = 0;
    endtask

    initial begin
        #1 rst_n = 0;
        @(negedge clk);
        rst_n = 1;
        for (int i = 0; i < 20; i++) send(1000 + i, i % 4);
        // Interrupt a live stream, including filter and detector state.
        @(negedge clk);
        rst_n = 0;
        #1;
        if (event_valid || event_overflow || dut.detector.completed_frames != 0 ||
            dut.merger.count != 0 || dut.filter.high_pass.state_storage.x1_mem[0] != 0 ||
            dut.filter.low_pass.state_storage.y1_mem[0] != 0)
            $fatal(1, "pipeline state not cleared by reset");
        @(negedge clk);
        rst_n = 1;
        send(1000, 0);
        wait (dut.filtered_valid);
        if (dut.filtered_code !== 24'sd179 || dut.filtered_channel !== 0)
            $fatal(1, "post-reset first sample wrong");
        $display("PASS: integrated reset clears state; first post-reset output is correct");
        $finish;
    end
endmodule
