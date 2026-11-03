`timescale 1ns/1ps

module neurosig_pipeline_reference_tb #(
    parameter int FRAMES = 32000,
    parameter int CHANNELS = 32,
    parameter int CHANNEL_BITS = (CHANNELS > 1 ? $clog2(CHANNELS) : 1)
);
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 1, flush = 0;
    logic sample_valid = 0, sample_ready;
    logic signed [15:0] sample_code = 0;
    logic [CHANNEL_BITS-1:0] channel_id = 0;
    logic [CHANNELS*24-1:0] threshold_codes = '0;
    logic event_valid, event_overflow;
    logic signed [23:0] event_code;
    logic [31:0] event_index;
    logic [CHANNEL_BITS-1:0] event_channel;
    logic [15:0] adc_bits;
    logic [23:0] threshold_bits;
    integer adc_fd, threshold_fd, events_fd, observed_fd;
    integer expected_count, observed_count = 0;
    integer expected_index, expected_channel, expected_code, scanned;
    string adc_path, threshold_path, events_path, observed_path;

    neurosig_pipeline #(.CHANNELS(CHANNELS), .CHANNEL_BITS(CHANNEL_BITS)) dut (.*);

    always @(posedge clk) begin
        #1;
        if (rst_n && event_valid) begin
            if (event_overflow) $fatal(1, "event queue overflow");
            if (observed_count >= expected_count)
                $fatal(1, "extra RTL event (%0d,%0d,%0d)",
                       event_index, event_channel, event_code);
            scanned = $fscanf(events_fd, "%d %d %d\n",
                              expected_index, expected_channel, expected_code);
            if (scanned != 3) $fatal(1, "missing Python event %0d", observed_count);
            if (event_index !== expected_index ||
                event_channel !== CHANNEL_BITS'(expected_channel) ||
                event_code !== 24'(expected_code))
                $fatal(1, "event %0d RTL=(%0d,%0d,%0d) Python=(%0d,%0d,%0d)",
                       observed_count, event_index, event_channel, event_code,
                       expected_index, expected_channel, expected_code);
            $fdisplay(observed_fd, "%0d %0d %0d", event_index, event_channel, event_code);
            observed_count++;
        end
    end

    initial begin
        #100000000;
        $fatal(1, "pipeline timeout after %0d events", observed_count);
    end

    initial begin
        if (!$value$plusargs("ADC=%s", adc_path) ||
            !$value$plusargs("THRESHOLDS=%s", threshold_path) ||
            !$value$plusargs("EVENTS=%s", events_path) ||
            !$value$plusargs("OBSERVED=%s", observed_path))
            $fatal(1, "pass +ADC=... +THRESHOLDS=... +EVENTS=... +OBSERVED=...");
        adc_fd = $fopen(adc_path, "r");
        threshold_fd = $fopen(threshold_path, "r");
        events_fd = $fopen(events_path, "r");
        observed_fd = $fopen(observed_path, "w");
        if (adc_fd == 0 || threshold_fd == 0 || events_fd == 0 || observed_fd == 0)
            $fatal(1, "cannot open vector files");
        scanned = $fscanf(events_fd, "%d\n", expected_count);
        if (scanned != 1) $fatal(1, "bad expected event count");
        for (int c = 0; c < CHANNELS; c++) begin
            scanned = $fscanf(threshold_fd, "%h\n", threshold_bits);
            if (scanned != 1) $fatal(1, "missing threshold %0d", c);
            threshold_codes[c*24 +: 24] = threshold_bits;
        end
        #1 rst_n = 0;
        @(negedge clk);
        rst_n = 1;
        for (int i = 0; i < FRAMES*CHANNELS; i++) begin
            scanned = $fscanf(adc_fd, "%h\n", adc_bits);
            if (scanned != 1) $fatal(1, "missing ADC code %0d", i);
            @(negedge clk);
            while (!sample_ready) @(negedge clk);
            sample_valid = 1;
            sample_code = adc_bits;
            channel_id = CHANNEL_BITS'(i % CHANNELS);
            @(posedge clk);
            #1;
            @(negedge clk);
            sample_valid = 0;
        end
        repeat (100) @(negedge clk);
        flush = 1;
        repeat (100) @(negedge clk);
        if (event_overflow) $fatal(1, "event queue overflow");
        if (observed_count != expected_count)
            $fatal(1, "RTL produced %0d events, Python produced %0d",
                   observed_count, expected_count);
        $display("PASS: %0d ADC samples, %0d exact merged events", FRAMES*CHANNELS,
                 observed_count);
        $fclose(adc_fd);
        $fclose(threshold_fd);
        $fclose(events_fd);
        $fclose(observed_fd);
        $finish;
    end
endmodule
