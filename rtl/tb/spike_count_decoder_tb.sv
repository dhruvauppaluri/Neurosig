`timescale 1ns/1ps

module spike_count_decoder_tb;
    logic [255:0] counts;
    logic [511:0] weights;
    logic signed [31:0] bias;
    logic signed [31:0] score;
    logic prediction;
    integer weights_fd, vectors_fd, scanned, weight, expected_score;
    integer checks = 0;
    integer expected_vectors = 1024;
    string weights_path, vectors_path;

    spike_count_decoder dut (.*);

    initial begin
        if (!$value$plusargs("WEIGHTS=%s", weights_path) ||
            !$value$plusargs("VECTORS=%s", vectors_path))
            $fatal(1, "pass +WEIGHTS=... +VECTORS=...");
        scanned = $value$plusargs("EXPECTED=%d", expected_vectors);
        weights_fd = $fopen(weights_path, "r");
        vectors_fd = $fopen(vectors_path, "r");
        if (weights_fd == 0 || vectors_fd == 0) $fatal(1, "cannot open decoder vectors");
        scanned = $fscanf(weights_fd, "%d\n", bias);
        if (scanned != 1) $fatal(1, "missing bias");
        for (int c = 0; c < 32; c++) begin
            scanned = $fscanf(weights_fd, "%d\n", weight);
            if (scanned != 1) $fatal(1, "missing weight %0d", c);
            weights[c*16 +: 16] = 16'(weight);
        end
        while (!$feof(vectors_fd)) begin
            scanned = $fscanf(vectors_fd, "%h %d\n", counts, expected_score);
            if (scanned == -1) break;
            if (scanned != 2) $fatal(1, "malformed vector %0d", checks);
            #1;
            if (score !== expected_score || prediction !== (expected_score >= 0))
                $fatal(1, "vector %0d: score %0d expected %0d", checks, score, expected_score);
            checks++;
        end
        if (checks != expected_vectors)
            $fatal(1, "expected %0d vectors, got %0d", expected_vectors, checks);
        $display("PASS: %0d exact decoder scores and predictions", checks);
        $finish;
    end
endmodule
