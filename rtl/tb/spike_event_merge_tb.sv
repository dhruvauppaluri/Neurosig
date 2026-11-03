`timescale 1ns/1ps

module spike_event_merge_tb;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 1, flush = 0;
    logic [31:0] completed_frames = 0;
    logic candidate_valid = 0;
    logic signed [23:0] candidate_code = 0;
    logic [31:0] candidate_index = 0;
    logic [4:0] candidate_channel = 0;
    logic signed [24:0] candidate_margin = 0;
    logic event_valid, overflow;
    logic signed [23:0] event_code;
    logic [31:0] event_index;
    logic [4:0] event_channel;

    spike_event_merge dut (.*);

    task automatic push(input int index, channel, margin, code);
        @(negedge clk);
        candidate_valid = 1;
        candidate_index = index;
        candidate_channel = channel;
        candidate_margin = margin;
        candidate_code = code;
        @(posedge clk);
        #1;
        @(negedge clk);
        candidate_valid = 0;
    endtask

    task automatic expect_event(input int index, channel, code);
        @(posedge clk);
        #1;
        if (!event_valid || event_index !== index ||
            event_channel !== channel || event_code !== code)
            $fatal(1, "got valid=%b index=%0d channel=%0d code=%0d; expected index=%0d channel=%0d code=%0d",
                   event_valid, event_index, event_channel, event_code,
                   index, channel, code);
    endtask

    initial begin
        #1 rst_n = 0;
        @(negedge clk);
        rst_n = 1;
        // Arrive out of sample-index order. 100 and 116 share a group;
        // 117 is outside the first group's anchored 16-sample window.
        push(116, 2, 30, -130);
        push(117, 3, 40, -140);
        push(100, 0, 10, -110);
        push(105, 1, 35, -135);
        @(negedge clk);
        completed_frames = 133;
        expect_event(105, 1, -135);
        @(negedge clk);
        flush = 1;
        candidate_valid = 1; // append while compacting the second group
        candidate_index = 200;
        candidate_channel = 4;
        candidate_margin = 5;
        candidate_code = -105;
        expect_event(117, 3, -140);
        @(negedge clk);
        candidate_valid = 0;
        expect_event(200, 4, -105);
        @(posedge clk);
        #1;
        if (event_valid || overflow) $fatal(1, "unexpected third event or overflow");
        @(negedge clk);
        rst_n = 0;
        flush = 0;
        completed_frames = 0;
        @(negedge clk);
        rst_n = 1;
        for (int i = 0; i < 65; i++) push(100, i % 32, i, -100);
        if (!overflow) $fatal(1, "65th candidate must set overflow for depth 64");
        @(negedge clk);
        rst_n = 0;
        #1;
        if (overflow) $fatal(1, "reset did not clear overflow");
        $display("PASS: spike_event_merge_tb (ordering, grouping, flush, concurrent append, overflow/reset)");
        $finish;
    end
endmodule
