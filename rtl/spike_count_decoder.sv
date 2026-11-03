`timescale 1ns/1ps

// One 32-channel count window -> binary synthetic-state prediction.
// Weights and bias are Q8.8; score remains in Q8.8 integer units.
module spike_count_decoder (
    input  logic [255:0] counts,
    input  logic [511:0] weights,
    input  logic signed [31:0] bias,
    output logic signed [31:0] score,
    output logic prediction
);
    always_comb begin
        score = bias;
        for (int c = 0; c < 32; c++)
            score += 32'($signed(weights[c*16 +: 16]) *
                         $signed({1'b0, counts[c*8 +: 8]}));
        prediction = score >= 0;
    end
endmodule
