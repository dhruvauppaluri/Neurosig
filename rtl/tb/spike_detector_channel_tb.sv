`timescale 1ns/1ps

module spike_detector_channel_tb;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 1;
    logic sample_valid = 0;
    logic signed [23:0] sample_code = 0;
    logic signed [23:0] threshold_code = -24'sd100;
    logic [31:0] sample_index = 0;
    logic spike_valid;
    logic signed [23:0] spike_code;
    logic [31:0] spike_index;
    int checks = 0;

    spike_detector_channel dut (.*);

    task automatic reset_dut;
        @(negedge clk);
        rst_n = 0;
        sample_valid = 0;
        #1;
        if (spike_valid !== 1'b0) $fatal(1, "spike valid during reset");
        @(negedge clk);
        rst_n = 1;
    endtask

    task automatic sample(input int index, input int code,
                          input bit expected_valid,
                          input int expected_code = 0,
                          input int expected_index = 0);
        @(negedge clk);
        sample_valid = 1;
        sample_index = index;
        sample_code = code;
        @(posedge clk);
        #1;
        if (spike_valid !== expected_valid)
            $fatal(1, "sample %0d: spike_valid=%b expected=%b", index,
                   spike_valid, expected_valid);
        if (expected_valid &&
            (spike_code !== expected_code || spike_index !== expected_index))
            $fatal(1, "sample %0d: event=(%0d,%0d) expected=(%0d,%0d)",
                   index, spike_code, spike_index, expected_code, expected_index);
        checks++;
        @(negedge clk);
        sample_valid = 0;
    endtask

    task automatic gap(input int cycles);
        repeat (cycles) begin
            @(posedge clk);
            #1;
            if (spike_valid !== 1'b0) $fatal(1, "spike during invalid clock");
            checks++;
        end
    endtask

    initial begin
        reset_dut();
        sample(0, -90, 0);
        sample(1, -110, 0); // downward crossing
        sample(2, -120, 0);
        sample(3, -150, 0); // earliest minimum
        gap(3);             // gaps must not advance the search
        for (int i = 4; i <= 16; i++) sample(i, -120, 0);
        sample(17, -150, 1, -150, 3); // tie keeps earliest index
        gap(2); // spike_valid is a pulse

        reset_dut();
        sample(0, -90, 0);
        sample(1, -110, 0);
        for (int i = 2; i <= 16; i++) sample(i, -120, 0);
        sample(17, -200, 1, -200, 17); // last sample must be included

        // A crossing 20 samples after the last peak is rejected.
        sample(18, -90, 0);
        sample(20, -110, 0);
        for (int i = 21; i <= 36; i++) sample(i, -90, 0);
        // Index 49 is 32 samples after peak 17; this crossing is accepted.
        sample(48, -90, 0);
        sample(49, -110, 0);
        for (int i = 50; i <= 64; i++) sample(i, -120, 0);
        sample(65, -130, 1, -130, 65);

        $display("PASS: spike_detector_channel_tb (%0d sample/idle checks)", checks);
        $finish;
    end
endmodule
