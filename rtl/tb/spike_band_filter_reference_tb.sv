`timescale 1ns/1ps

module spike_band_filter_reference_tb #(
    parameter bit PIPELINED = 0
);
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 1;
    logic in_valid = 0, in_ready;
    logic signed [23:0] in_sample = 0;
    logic [4:0] in_channel = 0;
    logic out_valid, out_ready = 0;
    logic signed [23:0] out_sample;
    logic [4:0] out_channel;
    integer fd, scanned, channel, vector_count = 0, stall;
    logic [15:0] input_bits;
    logic [23:0] expected_bits;
    string vector_path;

    spike_band_filter #(.CHANNELS(32), .PIPELINED(PIPELINED)) dut (.*);

    initial begin
        #10000000;
        $fatal(1, "timeout after %0d vectors", vector_count);
    end

    initial begin
        if (!$value$plusargs("VECTORS=%s", vector_path))
            $fatal(1, "pass +VECTORS=/path/to/vectors.txt");
        fd = $fopen(vector_path, "r");
        if (!fd) $fatal(1, "cannot open %s", vector_path);
        #1 rst_n = 0;
        @(negedge clk);
        rst_n = 1;

        while (!$feof(fd)) begin
            scanned = $fscanf(fd, "%d %h %h\n", channel, input_bits, expected_bits);
            if (scanned == -1) break;
            if (scanned != 3) $fatal(1, "malformed vector %0d", vector_count);
            if (channel < 0 || channel >= 32) $fatal(1, "invalid channel");
            @(negedge clk);
            while (!in_ready) @(negedge clk);
            in_valid = 1;
            in_channel = channel[4:0];
            in_sample = {{8{input_bits[15]}}, input_bits};
            @(posedge clk);
            #1;
            if (in_ready) $fatal(1, "input handshake did not advance controller");
            @(negedge clk);
            in_valid = 0;

            while (!out_valid) @(negedge clk);
            if (out_channel !== channel[4:0] || out_sample !== expected_bits)
                $fatal(1, "vector %0d channel %0d: got %0d (%h), expected %0d (%h)",
                       vector_count, channel, out_sample, out_sample,
                       $signed(expected_bits), expected_bits);
            stall = vector_count % 4;
            repeat (stall) begin
                @(posedge clk);
                #1;
                if (!out_valid || out_channel !== channel[4:0] ||
                    out_sample !== expected_bits)
                    $fatal(1, "vector %0d changed during backpressure", vector_count);
            end
            @(negedge clk);
            out_ready = 1;
            @(posedge clk);
            #1;
            if (out_valid) $fatal(1, "vector %0d was not consumed", vector_count);
            @(negedge clk);
            out_ready = 0;
            vector_count++;
        end
        $fclose(fd);
        if (!vector_count) $fatal(1, "no vectors read");
        $display("PASS: %0d Python-to-RTL filter vectors, exact codes and backpressure", vector_count);
        $finish;
    end
endmodule
