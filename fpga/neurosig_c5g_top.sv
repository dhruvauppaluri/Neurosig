`timescale 1ns/1ps

module neurosig_c5g_top(input logic CLOCK_50_B5B);
    logic [7:0] startup = 0;
    logic rst_n;
    logic [31:0] av_address, av_writedata, av_readdata;
    logic av_read, av_write, av_waitrequest, av_readdatavalid;
    logic [3:0] av_byteenable;
    logic av_master_reset;
    logic [31:0] slave_readdata, registered_readdata;

    always_ff @(posedge CLOCK_50_B5B) begin
        if (startup != 8'hff) startup <= startup + 1'b1;
        av_readdatavalid <= av_read;
        if (av_read) registered_readdata <= slave_readdata;
    end
    assign rst_n = startup == 8'hff;
    assign av_readdata = registered_readdata;

    neurosig_jtag jtag (
        .master_0_clk_clk(CLOCK_50_B5B),
        .master_0_clk_reset_reset(~rst_n),
        .master_0_master_address(av_address),
        .master_0_master_readdata(av_readdata),
        .master_0_master_read(av_read),
        .master_0_master_write(av_write),
        .master_0_master_writedata(av_writedata),
        .master_0_master_waitrequest(av_waitrequest),
        .master_0_master_readdatavalid(av_readdatavalid),
        .master_0_master_byteenable(av_byteenable),
        .master_0_master_reset_reset(av_master_reset)
    );

    neurosig_c5g_mailbox mailbox (
        .clk(CLOCK_50_B5B), .rst_n(rst_n),
        .address(av_address[13:2]), .read(av_read),
        .write(av_write && (&av_byteenable)), .writedata(av_writedata),
        .readdata(slave_readdata), .waitrequest(av_waitrequest)
    );
endmodule
