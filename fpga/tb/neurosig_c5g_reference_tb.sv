`timescale 1ns/1ps
module neurosig_c5g_reference_tb;
    localparam int CHANNELS=32, FRAMES=32000, BLOCK=1024, TOTAL=CHANNELS*FRAMES;
    logic clk=0;
    always #5 clk=~clk;
    logic rst_n=0, read=0, write=0;
    logic [11:0] address=0;
    logic [31:0] writedata=0, readdata;
    logic waitrequest;
    neurosig_c5g_mailbox dut (.*);
    integer adc_fd, thresholds_fd, events_fd, seen=0, expected_count, scan;
    integer expected_index, expected_channel, expected_code;
    logic [15:0] adc_bits;
    logic [23:0] threshold_bits;
    string adc_path, threshold_path, events_path;

    task automatic wr(input [11:0] a,input [31:0] d);
        @(negedge clk); address=a; writedata=d; write=1;
        @(negedge clk); write=0;
    endtask
    task automatic idle;
        while ((dut.state != dut.IDLE) || !rst_n) @(negedge clk);
    endtask
    function automatic [31:0] crc_byte(input [31:0] ci,input [7:0] b);
        logic [31:0] v;
        v=ci^b;
        for(int j=0;j<8;j++) v=v[0] ? (v>>1)^32'hedb88320 : v>>1;
        return v;
    endfunction
    function automatic [31:0] crc_word(input [31:0] ci,input [31:0] w);
        logic [31:0] v;
        v=ci;
        for(int j=0;j<4;j++) v=crc_byte(v,w[j*8 +: 8]);
        return v;
    endfunction
    logic [31:0] word, crc, data;
    integer cycles=0, stalls=0, max_merge_count=0;
    always @(posedge clk) if (rst_n) begin
        cycles <= cycles + 1;
        if (dut.sample_valid && !dut.sample_ready) stalls <= stalls + 1;
        if ($countones(dut.core.serial_merger.merger.occupied) > max_merge_count)
            max_merge_count <= $countones(dut.core.serial_merger.merger.occupied);
    end
    initial begin
        if (!$value$plusargs("ADC=%s",adc_path) ||
            !$value$plusargs("THRESHOLDS=%s",threshold_path) ||
            !$value$plusargs("EVENTS=%s",events_path)) $fatal(1,"missing vectors");
        adc_fd=$fopen(adc_path,"r");
        thresholds_fd=$fopen(threshold_path,"r");
        events_fd=$fopen(events_path,"r");
        if (!adc_fd || !thresholds_fd || !events_fd) $fatal(1,"cannot open vectors");
        scan=$fscanf(events_fd,"%d\n",expected_count);
        if (scan!=1) $fatal(1,"bad event count");
        repeat(3) @(negedge clk); rst_n=1;
        idle();
        wr(12'h004,32'd20261103);
        for(int c=0;c<CHANNELS;c++) begin
            scan=$fscanf(thresholds_fd,"%h\n",threshold_bits);
            if(scan!=1) $fatal(1,"missing threshold");
            wr(12'h040+c,{8'b0,threshold_bits});
        end
        for(int block=0;block<TOTAL/BLOCK;block++) begin
            crc=32'hffffffff;
            for(int i=0;i<BLOCK;i++) begin
                scan=$fscanf(adc_fd,"%h\n",adc_bits);
                if(scan!=1) $fatal(1,"missing ADC sample");
                word={11'b0,5'((block*BLOCK+i)%CHANNELS),adc_bits};
                wr(12'h400+i,word);
                crc=crc_word(crc,word);
            end
            wr(12'h005,32'(block));
            wr(12'h006,BLOCK);
            wr(12'h007,crc^32'hffffffff);
            wr(12'h010,32'h434f4d54);
            idle();
            if(dut.error_flags || dut.last_ack!=32'(block))
                $fatal(1,"block %0d failed: flags=%h ack=%h",block,dut.error_flags,dut.last_ack);
        end
        wr(12'h010,32'h464c5348);
        idle();
        if(dut.total_samples!=TOTAL || dut.error_flags || dut.core_overflow)
            $fatal(1,"sample/overflow failure samples=%0d flags=%h",dut.total_samples,dut.error_flags);
        while(dut.event_count!=0) begin
            address=12'h00b; #1;
            scan=$fscanf(events_fd,"%d %d %d\n",expected_index,expected_channel,expected_code);
            if(scan!=3 || readdata!==32'(expected_index)) $fatal(1,"index mismatch at %0d",seen);
            address=12'h00c; #1; data=readdata;
            if(data[29]!==1'b1 || data[28:24]!==5'(expected_channel) ||
               $signed(data[23:0])!==24'(expected_code))
                $fatal(1,"event mismatch at %0d: %h",seen,data);
            address=12'h00d; #1;
            if(readdata!==32'(seen)) $fatal(1,"sequence mismatch at %0d",seen);
            wr(12'h00f,1);
            seen++;
        end
        if(seen!=expected_count) $fatal(1,"saw %0d instead of %0d events",seen,expected_count);
        if (stalls == 0) $fatal(1,"ready/valid backpressure was not exercised");
        $display("PASS C5G mailbox: %0d accepted ADC samples, %0d exact events, %0d stalls, %0d total cycles, maximum merge occupancy %0d",
                 TOTAL,seen,stalls,cycles,max_merge_count);
        $finish;
    end
endmodule
