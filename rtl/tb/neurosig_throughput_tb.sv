`timescale 1ns/1ps

module neurosig_throughput_tb;
    localparam int CHANNELS = 32;
    localparam int FRAMES = 4096;
    localparam int TOTAL = CHANNELS * FRAMES;
    localparam real ASSUMED_CLOCK_HZ = 100_000_000.0;
    localparam real REQUIRED_SAMPLES_PER_SECOND = 32.0 * 32_000.0;

    logic clk = 0;
    always #5 clk = ~clk; // 10 ns period: assumed 100 MHz for rate conversion
    logic rst_n = 1, flush = 0;
    logic sample_valid = 0, sample_ready;
    logic signed [15:0] sample_code = 0;
    logic [4:0] channel_id = 0;
    logic [CHANNELS*24-1:0] threshold_codes = '0;
    logic event_valid, event_overflow;
    logic signed [23:0] event_code;
    logic [31:0] event_index;
    logic [4:0] event_channel;

    longint unsigned cycles = 0;
    longint unsigned first_cycle = 0, last_cycle = 0, previous_cycle = 0;
    longint unsigned max_acceptance_gap = 0;
    int accepted_count = 0;
    int report_fd;
    string report_path;
    real cycles_per_sample, accepted_per_second, minimum_clock_hz;

    neurosig_pipeline dut (.*);

    always @(posedge clk) begin
        if (rst_n) begin
            cycles++;
            if (event_valid || event_overflow)
                $fatal(1, "unexpected event or queue overflow on zero input");
            if (sample_valid && sample_ready) begin
                if (accepted_count == 0) first_cycle = cycles;
                else if (cycles - previous_cycle > max_acceptance_gap)
                    max_acceptance_gap = cycles - previous_cycle;
                previous_cycle = cycles;
                last_cycle = cycles;
                accepted_count++;
            end
        end
    end

    initial begin
        #100000000;
        $fatal(1, "throughput test timed out after %0d accepted samples", accepted_count);
    end

    initial begin
        if (!$value$plusargs("REPORT=%s", report_path))
            $fatal(1, "pass +REPORT=/path/to/report.json");
        for (int c = 0; c < CHANNELS; c++)
            threshold_codes[c*24 +: 24] = -24'sd100;
        #1 rst_n = 0;
        @(negedge clk);
        rst_n = 1;

        // Drive valid continuously. Only an actual rising-edge handshake
        // advances the sample and channel; data remains stable under stalls.
        while (accepted_count < TOTAL) begin
            @(negedge clk);
            sample_valid = 1;
            sample_code = 16'sd0;
            channel_id = 5'(accepted_count % CHANNELS);
        end
        sample_valid = 0;
        repeat (100) @(negedge clk);
        flush = 1;
        repeat (10) @(negedge clk);

        if (accepted_count != TOTAL || event_overflow)
            $fatal(1, "incomplete stream or queue overflow");
        cycles_per_sample = real'(last_cycle - first_cycle + 1) / TOTAL;
        accepted_per_second = ASSUMED_CLOCK_HZ / cycles_per_sample;
        minimum_clock_hz = REQUIRED_SAMPLES_PER_SECOND * cycles_per_sample;
        report_fd = $fopen(report_path, "w");
        if (report_fd == 0) $fatal(1, "cannot open throughput report");
        $fdisplay(report_fd, "{");
        $fdisplay(report_fd, "  \"accepted_samples\": %0d,", TOTAL);
        $fdisplay(report_fd, "  \"first_acceptance_cycle\": %0d,", first_cycle);
        $fdisplay(report_fd, "  \"last_acceptance_cycle\": %0d,", last_cycle);
        $fdisplay(report_fd, "  \"elapsed_cycles_inclusive\": %0d,", last_cycle-first_cycle+1);
        $fdisplay(report_fd, "  \"average_cycles_per_sample\": %0.6f,", cycles_per_sample);
        $fdisplay(report_fd, "  \"maximum_cycles_between_acceptances\": %0d,", max_acceptance_gap);
        $fdisplay(report_fd, "  \"assumed_clock_hz\": 100000000,");
        $fdisplay(report_fd, "  \"projected_samples_per_second_at_assumed_clock\": %0.3f,", accepted_per_second);
        $fdisplay(report_fd, "  \"required_samples_per_second\": 1024000,");
        $fdisplay(report_fd, "  \"minimum_clock_hz_for_required_average_rate\": %0.3f,", minimum_clock_hz);
        $fdisplay(report_fd, "  \"stimulus\": \"4096 frames x 32 channels, zero codes, valid held high\",");
        $fdisplay(report_fd, "  \"timing_qualification\": \"Assumed clock only; no synthesis or static timing verification\"");
        $fdisplay(report_fd, "}");
        $fclose(report_fd);
        $display("PASS: %0d accepted in %0d cycles; %.6f cycles/sample; max gap %0d cycles",
                 TOTAL, last_cycle-first_cycle+1, cycles_per_sample, max_acceptance_gap);
        $display("At assumed 100 MHz: %.3f samples/s; minimum clock for 1.024M/s: %.3f Hz",
                 accepted_per_second, minimum_clock_hz);
        $finish;
    end
endmodule
