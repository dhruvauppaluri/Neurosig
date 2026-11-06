`timescale 1ns/1ps

// 32-bit Avalon-MM slave. Addresses are word addresses (byte address >> 2).
// See fpga/README.md for the frozen register protocol.
module neurosig_c5g_mailbox #(
    parameter int CHANNELS = 32,
    parameter int BLOCK_WORDS = 1024,
    parameter int EVENT_DEPTH = 256
) (
    input logic clk,
    input logic rst_n,
    input logic [11:0] address,
    input logic read,
    input logic write,
    input logic [31:0] writedata,
    output logic [31:0] readdata,
    output logic waitrequest
);
    localparam int CHANNEL_BITS = (CHANNELS > 1 ? $clog2(CHANNELS) : 1);
    localparam int BLOCK_BITS = $clog2(BLOCK_WORDS + 1);
    localparam int FIFO_BITS = $clog2(EVENT_DEPTH + 1);
    localparam logic [31:0] MAGIC = 32'h4e534734; // NSG4
    localparam logic [31:0] VERSION = 32'h0001_0001;
    localparam logic [11:0] DATA_BASE = 12'h400;

    typedef enum logic [2:0] {IDLE, CHECK_FETCH, CHECK, FEED_FETCH,
                              FEED, DRAIN, FLUSH, RESETTING} state_t;
    state_t state;
    logic [31:0] sample_ram [0:BLOCK_WORDS-1];
    logic [23:0] thresholds [0:CHANNELS-1];
    logic [CHANNELS*24-1:0] threshold_bus;
    logic [31:0] event_index_ram [0:EVENT_DEPTH-1];
    logic [31:0] event_data_ram [0:EVENT_DEPTH-1];
    logic [31:0] event_seq_ram [0:EVENT_DEPTH-1];
    logic [$clog2(EVENT_DEPTH)-1:0] head, tail;
    logic [FIFO_BITS-1:0] event_count;
    logic [31:0] session_id, command_seq, expected_crc, crc_reg;
    logic [31:0] last_ack, total_samples, event_sequence;
    logic [31:0] feed_cycles, check_cycles, fifo_peak;
    logic [31:0] error_flags;
    logic [BLOCK_BITS-1:0] block_len, position;
    logic [7:0] drain_count;
    logic [11:0] flush_count;
    logic pipeline_rst_n;
    logic sample_valid, sample_ready, flush;
    logic signed [15:0] sample_code;
    logic [CHANNEL_BITS-1:0] channel_id;
    logic event_valid, core_overflow;
    logic signed [23:0] event_code;
    logic [31:0] event_index;
    logic [CHANNEL_BITS-1:0] event_channel;
    logic [31:0] sample_word;

    genvar c;
    generate for (c=0; c<CHANNELS; c=c+1) begin : pack_threshold
        assign threshold_bus[c*24 +: 24] = thresholds[c];
    end endgenerate
    // A registered read lets Quartus use a RAM port instead of a large
    // combinational mux and a duplicate bank of flip-flops.
    always_ff @(posedge clk)
        sample_word <= sample_ram[position[$clog2(BLOCK_WORDS)-1:0]];
    assign sample_code = sample_word[15:0];
    assign channel_id = sample_word[16 +: CHANNEL_BITS];
    assign sample_valid = (state == FEED);
    assign flush = (state == FLUSH);
    assign pipeline_rst_n = rst_n && (state != RESETTING);
    assign waitrequest = 1'b0;

    // The seeded 32-channel trace peaks at three queued merge candidates.
    // Keep capacity for one candidate from every channel on the C5G.
    neurosig_pipeline #(.CHANNELS(CHANNELS), .CHANNEL_BITS(CHANNEL_BITS),
                        .MERGE_DEPTH(32), .SERIAL_MERGE(1),
                        .PIPELINED_FILTER(1)) core (
        .clk(clk), .rst_n(pipeline_rst_n), .sample_valid(sample_valid),
        .sample_ready(sample_ready), .sample_code(sample_code), .channel_id(channel_id),
        .threshold_codes(threshold_bus), .flush(flush), .event_valid(event_valid),
        .event_code(event_code), .event_index(event_index),
        .event_channel(event_channel), .event_overflow(core_overflow)
    );

    function automatic [31:0] crc_byte(input [31:0] crc_in, input [7:0] value);
        logic [31:0] v;
        v = crc_in ^ {24'b0, value};
        for (int i=0; i<8; i++) v = v[0] ? (v >> 1) ^ 32'hedb88320 : (v >> 1);
        return v;
    endfunction
    function automatic [31:0] crc_word(input [31:0] crc_in, input [31:0] value);
        logic [31:0] v;
        v = crc_in;
        for (int i=0; i<4; i++) v = crc_byte(v, value[i*8 +: 8]);
        return v;
    endfunction

    always_comb begin
        readdata = 32'b0;
        if (address >= 12'h040 && address < 12'h040 + CHANNELS)
            readdata = {8'b0, thresholds[address - 12'h040]};
        else case (address)
            12'h000: readdata = MAGIC;
            12'h001: readdata = VERSION;
            12'h002: readdata = {28'b0, state == FLUSH, state == IDLE, event_count != 0, state != IDLE};
            12'h003: readdata = error_flags | {30'b0, core_overflow, 1'b0};
            12'h004: readdata = session_id;
            12'h005: readdata = command_seq;
            12'h006: readdata = {{(32-BLOCK_BITS){1'b0}}, block_len};
            12'h007: readdata = expected_crc;
            12'h008: readdata = last_ack;
            12'h009: readdata = total_samples;
            12'h00a: readdata = {{(32-FIFO_BITS){1'b0}}, event_count};
            12'h00b: readdata = (event_count != 0) ? event_index_ram[head] : 32'b0;
            12'h00c: readdata = (event_count != 0) ? event_data_ram[head] : 32'b0;
            12'h00d: readdata = (event_count != 0) ? event_seq_ram[head] : 32'b0;
            12'h00e: readdata = crc_reg ^ 32'hffffffff;
            12'h011: readdata = feed_cycles;
            12'h012: readdata = check_cycles;
            12'h013: readdata = fifo_peak;
            12'h014: readdata = event_sequence;
            default: readdata = 32'b0;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= RESETTING;
            session_id <= 0;
            command_seq <= 0;
            expected_crc <= 0;
            crc_reg <= 32'hffffffff;
            last_ack <= 32'hffffffff;
            total_samples <= 0;
            event_sequence <= 0;
            feed_cycles <= 0;
            check_cycles <= 0;
            fifo_peak <= 0;
            error_flags <= 0;
            block_len <= 0;
            position <= 0;
            drain_count <= 0;
            flush_count <= 0;
            head <= 0;
            tail <= 0;
            event_count <= 0;
            for (int c=0; c<CHANNELS; c++) thresholds[c] <= 0;
        end else begin
            if (event_valid && event_count < EVENT_DEPTH) begin
                event_index_ram[tail] <= event_index;
                event_data_ram[tail] <= {error_flags[0],
                                         error_flags[1] | core_overflow, 1'b1,
                                         {{(5-CHANNEL_BITS){1'b0}}, event_channel}, event_code};
                event_seq_ram[tail] <= event_sequence;
                tail <= tail + 1'b1;
                event_sequence <= event_sequence + 1'b1;
            end else if (event_valid) error_flags[0] <= 1'b1;
            if (core_overflow) error_flags[1] <= 1'b1;
            if (event_valid && event_count < EVENT_DEPTH && write && address == 12'h00f && writedata == 1 && event_count != 0)
                event_count <= event_count;
            else if (event_valid && event_count < EVENT_DEPTH) begin
                event_count <= event_count + 1'b1;
                if (event_count + 1'b1 > fifo_peak)
                    fifo_peak <= event_count + 1'b1;
            end
            else if (write && address == 12'h00f && writedata == 1 && event_count != 0) event_count <= event_count - 1'b1;

            if (state == FEED || state == FEED_FETCH) feed_cycles <= feed_cycles + 1'b1;
            if (state == CHECK || state == CHECK_FETCH) check_cycles <= check_cycles + 1'b1;

            if (write) begin
                if (address >= DATA_BASE && address < DATA_BASE + BLOCK_WORDS) begin
                    if (state == IDLE) sample_ram[address - DATA_BASE] <= writedata;
                    else error_flags[2] <= 1'b1;
                end else if (address >= 12'h040 && address < 12'h040 + CHANNELS) begin
                    if (state == IDLE && total_samples == 0) thresholds[address - 12'h040] <= writedata[23:0];
                    else error_flags[2] <= 1'b1;
                end else case (address)
                    12'h004: if (state == IDLE && total_samples == 0) session_id <= writedata;
                    12'h005: if (state == IDLE) command_seq <= writedata;
                    12'h006: if (state == IDLE) block_len <= BLOCK_BITS'(writedata);
                    12'h007: if (state == IDLE) expected_crc <= writedata;
                    12'h00f: if (writedata == 1 && event_count != 0) head <= head + 1'b1;
                    12'h010: begin
                        if (writedata == 32'h52455354 && state == IDLE) begin // REST
                            state <= RESETTING;
                            last_ack <= 32'hffffffff;
                            total_samples <= 0;
                            event_sequence <= 0;
                            feed_cycles <= 0;
                            check_cycles <= 0;
                            fifo_peak <= 0;
                            error_flags <= 0;
                            head <= 0;
                            tail <= 0;
                            event_count <= 0;
                        end else if (writedata == 32'h434f4d54 && state == IDLE) begin // COMT
                            if (block_len == 0 || block_len > BLOCK_WORDS ||
                                (command_seq != last_ack + 1 && command_seq != last_ack))
                                error_flags[3] <= 1'b1;
                            else if (command_seq != last_ack) begin
                                position <= 0;
                                crc_reg <= 32'hffffffff;
                                state <= CHECK_FETCH;
                            end
                        end else if (writedata == 32'h464c5348 && state == IDLE) begin // FLSH
                            drain_count <= 0;
                            state <= DRAIN;
                        end else error_flags[3] <= 1'b1;
                    end
                    default: error_flags[3] <= 1'b1;
                endcase
            end

            case (state)
                IDLE: ;
                RESETTING: state <= IDLE;
                CHECK_FETCH: state <= CHECK;
                CHECK: begin
                    crc_reg <= crc_word(crc_reg, sample_word);
                    if (int'(sample_word[16 +: CHANNEL_BITS]) != ((total_samples + position) % CHANNELS))
                        error_flags[4] <= 1'b1;
                    if (position == block_len - 1'b1) begin
                        position <= 0;
                        if ((crc_word(crc_reg, sample_word) ^ 32'hffffffff) != expected_crc ||
                            error_flags[4] ||
                            (int'(sample_word[16 +: CHANNEL_BITS]) != ((total_samples + position) % CHANNELS))) begin
                            error_flags[5] <= 1'b1;
                            state <= IDLE;
                        end else state <= FEED_FETCH;
                    end else begin
                        position <= position + 1'b1;
                        state <= CHECK_FETCH;
                    end
                end
                FEED_FETCH: state <= FEED;
                FEED: if (sample_ready) begin
                    total_samples <= total_samples + 1'b1;
                    if (position == block_len - 1'b1) begin
                        last_ack <= command_seq;
                        state <= IDLE;
                    end else begin
                        position <= position + 1'b1;
                        state <= FEED_FETCH;
                    end
                end
                DRAIN: if (drain_count == 8'd32) begin
                    flush_count <= 0;
                    state <= FLUSH;
                end else drain_count <= drain_count + 1'b1;
                FLUSH: if (flush_count == 12'd4095) state <= IDLE;
                       else flush_count <= flush_count + 1'b1;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
