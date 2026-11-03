`timescale 1ns/1ps

module spike_detector_channel_reference_tb;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 1;
    logic sample_valid = 0;
    logic signed [23:0] sample_code = 0;
    logic signed [23:0] threshold_code;
    logic [31:0] sample_index = 0;
    logic spike_valid;
    logic signed [23:0] spike_code;
    logic [31:0] spike_index;
    integer trace_fd, events_fd, samples, expected_count, expected_index;
    integer expected_code, scanned, observed_count = 0;
    logic [23:0] input_bits;
    string trace_path, events_path;

    spike_detector_channel dut (.*);

    initial begin
        #1000000;
        $fatal(1, "detector reference test timed out");
    end

    initial begin
        if (!$value$plusargs("TRACE=%s", trace_path) ||
            !$value$plusargs("EVENTS=%s", events_path))
            $fatal(1, "pass +TRACE=... +EVENTS=...");
        trace_fd = $fopen(trace_path, "r");
        events_fd = $fopen(events_path, "r");
        if (!trace_fd || !events_fd) $fatal(1, "cannot open input files");
        scanned = $fscanf(trace_fd, "%d %d\n", threshold_code, samples);
        if (scanned != 2) $fatal(1, "bad trace header");
        scanned = $fscanf(events_fd, "%d\n", expected_count);
        if (scanned != 1) $fatal(1, "bad events header");
        #1 rst_n = 0;
        @(negedge clk);
        rst_n = 1;

        for (int i = 0; i < samples; i++) begin
            scanned = $fscanf(trace_fd, "%h\n", input_bits);
            if (scanned != 1) $fatal(1, "missing trace sample %0d", i);
            @(negedge clk);
            sample_valid = 1;
            sample_index = i;
            sample_code = input_bits;
            @(posedge clk);
            #1;
            if (spike_valid) begin
                if (observed_count >= expected_count)
                    $fatal(1, "extra RTL event at sample %0d", i);
                scanned = $fscanf(events_fd, "%d %d\n", expected_index, expected_code);
                if (scanned != 2) $fatal(1, "missing Python event %0d", observed_count);
                if (spike_index !== expected_index || spike_code !== expected_code)
                    $fatal(1, "event %0d: RTL (%0d,%0d), Python (%0d,%0d)",
                           observed_count, spike_index, spike_code,
                           expected_index, expected_code);
                observed_count++;
            end
            if (i % 11 == 0) begin
                @(negedge clk);
                sample_valid = 0;
                @(posedge clk);
                #1;
                if (spike_valid) $fatal(1, "event during invalid clock");
            end
        end
        @(negedge clk);
        sample_valid = 0;
        if (observed_count != expected_count)
            $fatal(1, "RTL produced %0d events; Python produced %0d",
                   observed_count, expected_count);
        $display("PASS: %0d samples, %0d exact Python-to-RTL events", samples, observed_count);
        $fclose(trace_fd);
        $fclose(events_fd);
        $finish;
    end
endmodule
