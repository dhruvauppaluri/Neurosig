`timescale 1ns/1ps
module neurosig_c5g_mailbox_tb;
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 0, read = 0, write = 0;
    logic [11:0] address = 0;
    logic [31:0] writedata = 0, readdata;
    logic waitrequest;
    neurosig_c5g_mailbox #(.CHANNELS(4), .BLOCK_WORDS(64), .EVENT_DEPTH(4)) dut (.*);

    task automatic wr(input [11:0] a, input [31:0] d);
        @(negedge clk); address=a; writedata=d; write=1;
        @(negedge clk); write=0;
    endtask
    task automatic check(input [11:0] a, input [31:0] expected);
        address=a; #1;
        if (readdata !== expected) $fatal(1,"reg %h: got %h expected %h",a,readdata,expected);
    endtask
    function automatic [31:0] crc_byte(input [31:0] crc_in, input [7:0] b);
        logic [31:0] v;
        v=crc_in ^ b;
        for (int i=0;i<8;i++) v=v[0] ? (v>>1)^32'hedb88320 : v>>1;
        return v;
    endfunction
    function automatic [31:0] crc_word(input [31:0] crc_in, input [31:0] w);
        logic [31:0] v;
        v=crc_in;
        for (int i=0;i<4;i++) v=crc_byte(v,w[i*8 +: 8]);
        return v;
    endfunction
    logic [31:0] crc;
    initial begin
        repeat(3) @(negedge clk); rst_n=1;
        repeat(2) @(negedge clk);
        check(12'h000,32'h4e534734);
        wr(12'h004,32'd123);
        for (int c=0;c<4;c++) wr(12'h040+c,32'hff0000);
        crc=32'hffffffff;
        for (int i=0;i<16;i++) begin
            wr(12'h400+i,32'((i%4)<<16));
            crc=crc_word(crc,32'((i%4)<<16));
        end
        wr(12'h005,0); wr(12'h006,16); wr(12'h007,crc^32'hffffffff);
        wr(12'h010,32'h434f4d54);
        repeat(150) @(negedge clk);
        check(12'h003,0);
        check(12'h008,0);
        check(12'h009,16);
        address=12'h011; #1;
        if (readdata < 16) $fatal(1,"feed cycle counter missing");
        check(12'h012,32);
        wr(12'h010,32'h434f4d54); // exact duplicate must not replay
        repeat(10) @(negedge clk);
        check(12'h009,16);
        // CRC corruption is rejected before any sample enters the core.
        wr(12'h005,1); wr(12'h007,0); wr(12'h010,32'h434f4d54);
        repeat(60) @(negedge clk);
        check(12'h009,16);
        address=12'h003; #1;
        if ((readdata & 32'h20) == 0) $fatal(1,"CRC error flag missing");
        wr(12'h010,32'h52455354);
        repeat(3) @(negedge clk);
        check(12'h003,0);
        // A valid CRC with the wrong channel must still be rejected before feed.
        wr(12'h400,32'h00010000);
        crc=crc_word(32'hffffffff,32'h00010000)^32'hffffffff;
        wr(12'h005,0); wr(12'h006,1); wr(12'h007,crc);
        wr(12'h010,32'h434f4d54);
        repeat(12) @(negedge clk);
        check(12'h009,0);
        address=12'h003; #1;
        if ((readdata & 32'h30) != 32'h30) $fatal(1,"channel order rejection missing");
        wr(12'h010,32'h52455354);
        repeat(3) @(negedge clk);
        // Inject a core output pulse to test lossless event buffering in isolation.
        force dut.event_valid=1;
        force dut.event_index=32'd42;
        force dut.event_channel=2'd3;
        force dut.event_code=-24'sd70000;
        @(negedge clk);
        release dut.event_valid;
        release dut.event_index;
        release dut.event_channel;
        release dut.event_code;
        check(12'h00a,1);
        check(12'h013,1);
        check(12'h014,1);
        check(12'h00b,42);
        address=12'h00c; #1;
        if (readdata[29] != 1 || readdata[28:24] != 3 ||
            $signed(readdata[23:0]) != -24'sd70000)
            $fatal(1,"event serialization lost 24-bit code or channel");
        check(12'h00d,0);
        wr(12'h00f,1);
        check(12'h00a,0);
        for (int i=0;i<5;i++) begin
            force dut.event_valid=1;
            force dut.event_index=32'd43;
            force dut.event_channel=2'd3;
            force dut.event_code=-24'sd70000;
            @(negedge clk);
            release dut.event_valid;
            release dut.event_index;
            release dut.event_channel;
            release dut.event_code;
            @(negedge clk);
        end
        check(12'h00a,4);
        check(12'h013,4);
        address=12'h003; #1;
        if (!readdata[0]) $fatal(1,"event FIFO overflow flag missing");
        $display("PASS mailbox transaction, CRC, ordering, event serialization, FIFO overflow");
        $finish;
    end
endmodule
