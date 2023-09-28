/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

// default_nettype of none prevents implicit wire declaration.
`default_nettype none
`timescale 1 ns / 1 ps 

`include "all_params.sv"

module krnl_vadd_rtl_int #( 
  parameter integer  C_S_AXI_CONTROL_DATA_WIDTH = 32,
  parameter integer  C_S_AXI_CONTROL_ADDR_WIDTH = 6,
  parameter integer  C_M_AXI_GMEM_ID_WIDTH      = 1,
  parameter integer  C_M_AXI_GMEM_ADDR_WIDTH    = 64,
  parameter integer  C_M_AXI_GMEM_DATA_WIDTH    = 512   // Data width of both input and output data
)
(
  // System signals
  input  wire  ap_clk,
  input  wire  ap_rst_n,
  // AXI4 master interface 
  output wire                                 m_axi_gmem_AWVALID,
  input  wire                                 m_axi_gmem_AWREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem_AWADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_gmem_AWID,
  output wire [7:0]                           m_axi_gmem_AWLEN,
  output wire [2:0]                           m_axi_gmem_AWSIZE,
  // Tie-off AXI4 transaction options that are not being used.
  output wire [1:0]                           m_axi_gmem_AWBURST,
  output wire [1:0]                           m_axi_gmem_AWLOCK,
  output wire [3:0]                           m_axi_gmem_AWCACHE,
  output wire [2:0]                           m_axi_gmem_AWPROT,
  output wire [3:0]                           m_axi_gmem_AWQOS,
  output wire [3:0]                           m_axi_gmem_AWREGION,
  output wire                                 m_axi_gmem_WVALID,
  input  wire                                 m_axi_gmem_WREADY,
  output wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_gmem_WDATA,
  output wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_gmem_WSTRB,
  output wire                                 m_axi_gmem_WLAST,
  output wire                                 m_axi_gmem_ARVALID,
  input  wire                                 m_axi_gmem_ARREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem_ARADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH-1:0]     m_axi_gmem_ARID,
  output wire [7:0]                           m_axi_gmem_ARLEN,
  output wire [2:0]                           m_axi_gmem_ARSIZE,
  output wire [1:0]                           m_axi_gmem_ARBURST,
  output wire [1:0]                           m_axi_gmem_ARLOCK,
  output wire [3:0]                           m_axi_gmem_ARCACHE,
  output wire [2:0]                           m_axi_gmem_ARPROT,
  output wire [3:0]                           m_axi_gmem_ARQOS,
  output wire [3:0]                           m_axi_gmem_ARREGION,
  input  wire                                 m_axi_gmem_RVALID,
  output wire                                 m_axi_gmem_RREADY,
  input  wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_gmem_RDATA,
  input  wire                                 m_axi_gmem_RLAST,
  input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_gmem_RID,
  input  wire [1:0]                           m_axi_gmem_RRESP,
  input  wire                                 m_axi_gmem_BVALID,
  output wire                                 m_axi_gmem_BREADY,
  input  wire [1:0]                           m_axi_gmem_BRESP,
  input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_gmem_BID,

  // AXI4-Lite slave interface
  input  wire                                    s_axi_control_AWVALID,
  output wire                                    s_axi_control_AWREADY,
  input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_AWADDR,
  input  wire                                    s_axi_control_WVALID,
  output wire                                    s_axi_control_WREADY,
  input  wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_WDATA,
  input  wire [C_S_AXI_CONTROL_DATA_WIDTH/8-1:0] s_axi_control_WSTRB,
  input  wire                                    s_axi_control_ARVALID,
  output wire                                    s_axi_control_ARREADY,
  input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_ARADDR,
  output wire                                    s_axi_control_RVALID,
  input  wire                                    s_axi_control_RREADY,
  output wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_RDATA,
  output wire [1:0]                              s_axi_control_RRESP,
  output wire                                    s_axi_control_BVALID,
  input  wire                                    s_axi_control_BREADY,
  output wire [1:0]                              s_axi_control_BRESP,
  output wire                                    interrupt 
);
///////////////////////////////////////////////////////////////////////////////
// Local Parameters (constants)
///////////////////////////////////////////////////////////////////////////////
//localparam integer LP_NUM_READ_CHANNELS  = 2;
// TODO: OPTIMIZE THESE PARAMETERS FOR OVERALL BANDWIDTH
localparam integer LP_NUM_READ_CHANNELS  = 1;
localparam integer LP_LENGTH_WIDTH       = 32;
localparam integer C_MAX_LENGTH_WIDTH    = LP_LENGTH_WIDTH;
localparam integer LP_DW_BYTES           = C_M_AXI_GMEM_DATA_WIDTH/8;
localparam integer LP_AXI_BURST_LEN      = 4096/LP_DW_BYTES < 256 ? 4096/LP_DW_BYTES : 256;
localparam integer LP_LOG_BURST_LEN      = $clog2(LP_AXI_BURST_LEN);
localparam integer LP_RD_MAX_OUTSTANDING = 3;
localparam integer LP_RD_FIFO_DEPTH      = LP_AXI_BURST_LEN*(LP_RD_MAX_OUTSTANDING + 1);
localparam integer LP_WR_FIFO_DEPTH      = LP_AXI_BURST_LEN;


///////////////////////////////////////////////////////////////////////////////
// Variables
///////////////////////////////////////////////////////////////////////////////
logic areset = 1'b0;  
logic ap_start;
logic ap_start_pulse;
logic ap_start_r;
logic ap_ready;
logic ap_done;
logic write_done;
logic ap_idle = 1'b1;
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] a;
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] b;
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] c;
logic [LP_LENGTH_WIDTH-1:0]         length_r;

logic read_done;
logic [LP_NUM_READ_CHANNELS-1:0] rd_tvalid;
logic [LP_NUM_READ_CHANNELS-1:0] rd_tready_n; 
logic [LP_NUM_READ_CHANNELS-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_tdata;
logic [LP_NUM_READ_CHANNELS-1:0] ctrl_rd_fifo_prog_full;
logic [LP_NUM_READ_CHANNELS-1:0] rd_fifo_tvalid_n;
logic [LP_NUM_READ_CHANNELS-1:0] rd_fifo_tready; 
logic [LP_NUM_READ_CHANNELS-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_fifo_tdata;

logic                               adder_tvalid;
logic                               adder_tready_n; 
logic [C_M_AXI_GMEM_DATA_WIDTH-1:0] adder_tdata;
logic                               wr_fifo_tvalid_n;
logic                               wr_fifo_tready; 
logic [C_M_AXI_GMEM_DATA_WIDTH-1:0] wr_fifo_tdata;

logic                               finished;
reg                                 finished_r;

reg                                 ap_done_r;

logic [C_MAX_LENGTH_WIDTH-1:0]      num_out_tx;
logic                               out_pulse;

logic                               trig_out;
logic                               trig_out_ack; 

reg [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   c_reg;
reg [LP_LENGTH_WIDTH-1:0]           length_r_reg;

///////////////////////////////////////////////////////////////////////////////
// RTL Logic 
///////////////////////////////////////////////////////////////////////////////
// Tie-off unused AXI protocol features
assign m_axi_gmem_AWID     = {C_M_AXI_GMEM_ID_WIDTH{1'b0}};
assign m_axi_gmem_AWBURST  = 2'b01;
assign m_axi_gmem_AWLOCK   = 2'b00;
assign m_axi_gmem_AWCACHE  = 4'b0011;
assign m_axi_gmem_AWPROT   = 3'b000;
assign m_axi_gmem_AWQOS    = 4'b0000;
assign m_axi_gmem_AWREGION = 4'b0000;
assign m_axi_gmem_ARBURST  = 2'b01;
assign m_axi_gmem_ARLOCK   = 2'b00;
assign m_axi_gmem_ARCACHE  = 4'b0011;
assign m_axi_gmem_ARPROT   = 3'b000;
assign m_axi_gmem_ARQOS    = 4'b0000;
assign m_axi_gmem_ARREGION = 4'b0000;

// Register and invert reset signal for better timing.
always @(posedge ap_clk) begin 
  areset <= ~ap_rst_n; 
end

// create pulse when ap_start transitions to 1
always @(posedge ap_clk) begin 
  begin 
    ap_start_r <= ap_start;
  end
end

assign ap_start_pulse = ap_start & ~ap_start_r;

// ap_idle is asserted when done is asserted, it is de-asserted when ap_start_pulse 
// is asserted
always @(posedge ap_clk) begin 
  if (areset) begin 
    ap_idle <= 1'b1;
  end
  else begin 
    ap_idle <= ap_done        ? 1'b1 : 
               ap_start_pulse ? 1'b0 : 
                                ap_idle;
    ap_done_r <= ap_done;
  end
end

//assign ap_done = finished & write_done;
assign ap_ready = ap_done;

// AXI4-Lite slave
krnl_vadd_rtl_control_s_axi #(
  .C_S_AXI_ADDR_WIDTH( C_S_AXI_CONTROL_ADDR_WIDTH ),
  .C_S_AXI_DATA_WIDTH( C_S_AXI_CONTROL_DATA_WIDTH )
) 
inst_krnl_vadd_control_s_axi (
  .AWVALID   ( s_axi_control_AWVALID         ) ,
  .AWREADY   ( s_axi_control_AWREADY         ) ,
  .AWADDR    ( s_axi_control_AWADDR          ) ,
  .WVALID    ( s_axi_control_WVALID          ) ,
  .WREADY    ( s_axi_control_WREADY          ) ,
  .WDATA     ( s_axi_control_WDATA           ) ,
  .WSTRB     ( s_axi_control_WSTRB           ) ,
  .ARVALID   ( s_axi_control_ARVALID         ) ,
  .ARREADY   ( s_axi_control_ARREADY         ) ,
  .ARADDR    ( s_axi_control_ARADDR          ) ,
  .RVALID    ( s_axi_control_RVALID          ) ,
  .RREADY    ( s_axi_control_RREADY          ) ,
  .RDATA     ( s_axi_control_RDATA           ) ,
  .RRESP     ( s_axi_control_RRESP           ) ,
  .BVALID    ( s_axi_control_BVALID          ) ,
  .BREADY    ( s_axi_control_BREADY          ) ,
  .BRESP     ( s_axi_control_BRESP           ) ,
  .ACLK      ( ap_clk                        ) ,
  .ARESET    ( areset                        ) ,
  .ACLK_EN   ( 1'b1                          ) ,
  .ap_start  ( ap_start                      ) ,
  .interrupt ( interrupt                     ) ,
  .ap_ready  ( ap_ready                      ) ,
  .ap_done   ( ap_done                       ) ,
  .ap_idle   ( ap_idle                       ) ,

  // The following signals expose scalar values
  // passed by the XRT OpenCL application.
  // a:        pointer to first input array
  // a:        pointer to second input array
  // a:        pointer to output array
  // length_r: number of bytes to read from
  //           each array and write back to
  //           array c.
  .a         ( a[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  .b         ( b[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  .c         ( c[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  .length_r  ( length_r[0+:LP_LENGTH_WIDTH]  ) 
);

always @(posedge ap_clk) begin 
  if (ap_start)
  begin
    c_reg <= c;
    length_r_reg <= length_r;
  end
end

// AXI4 Read Master
krnl_vadd_rtl_axi_read_master #( 
  .C_ADDR_WIDTH       ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
  .C_DATA_WIDTH       ( C_M_AXI_GMEM_DATA_WIDTH ) ,
  .C_ID_WIDTH         ( C_M_AXI_GMEM_ID_WIDTH   ) ,
  .C_NUM_CHANNELS     ( LP_NUM_READ_CHANNELS    ) ,
  .C_LENGTH_WIDTH     ( LP_LENGTH_WIDTH         ) ,

  // This controls the number of beats to be read
  // during each burst. 
  .C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
  .C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) ,
  .C_MAX_OUTSTANDING  ( LP_RD_MAX_OUTSTANDING   )
)
inst_axi_read_master ( 
  .aclk           ( ap_clk                 ) ,
  .areset         ( areset                 ) ,

  .ctrl_start     ( ap_start_pulse         ) ,
  .ctrl_done      ( read_done              ) ,
  //.ctrl_offset    ( {b,a}                  ) ,
  .ctrl_offset    ( {a}                      ) ,
  .ctrl_length    ( length_r               ) ,
  .ctrl_prog_full ( ctrl_rd_fifo_prog_full ) ,

  .arvalid        ( m_axi_gmem_ARVALID     ) ,
  .arready        ( m_axi_gmem_ARREADY     ) ,
  .araddr         ( m_axi_gmem_ARADDR      ) ,
  .arlen          ( m_axi_gmem_ARLEN       ) ,

  .rvalid         ( m_axi_gmem_RVALID      ) ,
  .rready         ( m_axi_gmem_RREADY      ) ,
  .rdata          ( m_axi_gmem_RDATA       ) ,
  .rlast          ( m_axi_gmem_RLAST       ) ,

  .rid            ( m_axi_gmem_RID         ) ,
  .rresp          ( m_axi_gmem_RRESP       ) ,

  .m_tvalid       ( rd_tvalid              ) ,
  .m_tready       ( ~rd_tready_n           ) ,
  .m_tdata        ( rd_tdata               ) ,

  .arid           ( m_axi_gmem_ARID        ) ,

  .arsize         ( m_axi_gmem_ARSIZE      )
);

/*
alt_read_axi_master #(
  .C_M_AXI_ADDR_WIDTH  ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
  .C_M_AXI_DATA_WIDTH  ( C_M_AXI_GMEM_DATA_WIDTH ) ,
  .C_XFER_SIZE_WIDTH   ( LP_LENGTH_WIDTH         ) ,
  .C_MAX_OUTSTANDING   ( LP_RD_MAX_OUTSTANDING   ) ,
  .C_INCLUDE_DATA_FIFO ( 1                       )
)
inst_alt_read_axi_master (
  .aclk                    ( ap_clk              ) ,
  .areset                  ( areset              ) ,

  .ctrl_start              ( ap_start_pulse      ) ,
  .ctrl_done               ( read_done           ) ,
  .ctrl_addr_offset        ( a                   ) ,
  .ctrl_xfer_size_in_bytes ( length_r << 6       ) ,

  .m_axi_arvalid           ( m_axi_gmem_ARVALID  ) ,
  .m_axi_arready           ( m_axi_gmem_ARREADY  ) ,
  .m_axi_araddr            ( m_axi_gmem_ARADDR   ) ,
  .m_axi_arlen             ( m_axi_gmem_ARLEN    ) ,

  .m_axi_rvalid            ( m_axi_gmem_RVALID   ) ,
  .m_axi_rready            ( m_axi_gmem_RREADY   ) ,
  .m_axi_rdata             ( m_axi_gmem_RDATA    ) ,
  .m_axi_rlast             ( m_axi_gmem_RLAST    ) ,

  .m_axis_aclk             ( ap_clk              ) ,
  .m_axis_areset           ( areset              ) ,
  .m_axis_tvalid           ( rd_tvalid           ) ,
  .m_axis_tready           ( ~rd_tready_n        ) ,
  .m_axis_tlast            (                     ) ,
  .m_axis_tdata            ( rd_tdata            ) ,

  .arid                    ( m_axi_gmem_ARID     ) ,

  .arsize                  ( m_axi_gmem_ARSIZE   )
);
*/

// xpm_fifo_sync: Synchronous FIFO
// Xilinx Parameterized Macro, Version 2016.4
xpm_fifo_sync # (
  .FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
  .ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
  .FIFO_WRITE_DEPTH          (LP_RD_FIFO_DEPTH),   //positive integer
  .WRITE_DATA_WIDTH          (C_M_AXI_GMEM_DATA_WIDTH),        //positive integer
  .WR_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),       //positive integer, Not used
  .PROG_FULL_THRESH          (LP_AXI_BURST_LEN-2),               //positive integer
  .FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
  .READ_MODE                 ("fwft"),            //string; "std" or "fwft";
  //.READ_MODE                 ("std"),            //string; "std" or "fwft";
  .FIFO_READ_LATENCY         (1),                //positive integer;
  //.FIFO_READ_LATENCY         (3),                //positive integer;
  .READ_DATA_WIDTH           (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .RD_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),               //positive integer, not used
  .PROG_EMPTY_THRESH         (10),               //positive integer, not used 
  .DOUT_RESET_VALUE          ("0"),              //string, don't care
  .WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

) inst_rd_xpm_fifo_sync[LP_NUM_READ_CHANNELS-1:0] (
  .sleep         ( 1'b0                    ) ,
  .rst           ( areset           ) ,
  //.rst           ( areset | ap_start_pulse | ap_done_r ),
  .wr_clk        ( ap_clk                  ) ,
  .wr_en         ( rd_tvalid               ) ,
  .din           ( rd_tdata                ) ,
  .full          ( rd_tready_n             ) ,
  .prog_full     ( ctrl_rd_fifo_prog_full  ) ,
  .wr_data_count (                         ) ,
  .overflow      (                         ) ,
  .wr_rst_busy   (                         ) ,
  .rd_en         ( rd_fifo_tready          ) ,
  .dout          ( rd_fifo_tdata           ) ,
  .empty         ( rd_fifo_tvalid_n        ) ,
  .prog_empty    (                         ) ,
  .rd_data_count (                         ) ,
  .underflow     (                         ) ,
  .rd_rst_busy   (                         ) ,
  .injectsbiterr ( 1'b0                    ) ,
  .injectdbiterr ( 1'b0                    ) ,
  .sbiterr       (                         ) ,
  .dbiterr       (                         ) 

);

/*
// Combinatorial Adder
krnl_vadd_rtl_adder #( 
  .C_DATA_WIDTH   ( C_M_AXI_GMEM_DATA_WIDTH ) ,
  .C_NUM_CHANNELS ( LP_NUM_READ_CHANNELS    ) 
)
inst_adder ( 
  .aclk     ( ap_clk            ) ,
  .areset   ( areset            ) ,

  .s_tvalid ( ~rd_fifo_tvalid_n ) ,
  .s_tready ( rd_fifo_tready    ) ,
  .s_tdata  ( rd_fifo_tdata     ) ,

  .m_tvalid ( adder_tvalid      ) ,
  .m_tready ( ~adder_tready_n   ) ,
  .m_tdata  ( adder_tdata       ) 
);
*/

multi_core_axi_enabled_cam #(
    .NUM_ROUTER_OUTPUTS(TOP_NUM_ROUTER_OUTPUTS),
    .NUM_MODE_CHANNELS(NUM_CORES)
) inst_multi_core_axi_enabled_cam (
    .clk            (ap_clk),
    .rst            (areset | ap_done_r),
    //.rst            (areset | ap_start_pulse | ap_done_r),
    //.rst            (areset | ap_start_pulse),

    .s_axis_tvalid  (~rd_fifo_tvalid_n[0]),
    .s_axis_tready  (rd_fifo_tready[0]),
    .s_axis_tdata   (rd_fifo_tdata[0]),

    .m_axis_tvalid  (adder_tvalid),
    .m_axis_tready  (~adder_tready_n),
    .m_axis_tdata   (adder_tdata),

    .num_out_tx     (num_out_tx),
    .out_pulse      (out_pulse),
    .finished       (finished),
   
    .trig_out       (trig_out),
    .trig_out_ack   (trig_out_ack) 
);

// We don't use the second channel,
// so we can just discard all data
// coming from it.
//assign rd_fifo_tready[1] = rd_fifo_tready[0];
//assign rd_fifo_tready[1] = 1;

/*
// ILA monitoring combinatorial adder
ila_0 i_ila_0 (
    .clk(ap_clk),              // input wire        clk
    .probe0(areset | ap_start_pulse | ap_done_r),           // input wire [0:0]  probe0  
    .probe1(rd_fifo_tvalid_n), // input wire [0:0]  probe1 
    .probe2(rd_fifo_tready),   // input wire [0:0]  probe2 
    //.probe3(rd_fifo_tdata),    // input wire [63:0] probe3
    .probe3(rd_fifo_tdata[0]),    // input wire [511:0] probe3  
    .probe4(adder_tvalid),     // input wire [0:0]  probe4 
    .probe5(adder_tready_n),   // input wire [0:0]  probe5 
    .probe6(adder_tdata[15:0]),       // input wire [511:0] probe6
    .trig_in(trig_out),
    .trig_in_ack(trig_out_ack)
);
*/

// xpm_fifo_sync: Synchronous FIFO
// Xilinx Parameterized Macro, Version 2016.4
xpm_fifo_sync # (
  .FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
  .ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
  .FIFO_WRITE_DEPTH          (LP_WR_FIFO_DEPTH),   //positive integer
  .WRITE_DATA_WIDTH          (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .WR_DATA_COUNT_WIDTH       ($clog2(LP_WR_FIFO_DEPTH)),               //positive integer, Not used
  .PROG_FULL_THRESH          (10),               //positive integer, Not used 
  .FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
  .READ_MODE                 ("fwft"),            //string; "std" or "fwft";
  //.READ_MODE                 ("std"),            //string; "std" or "fwft";
  .FIFO_READ_LATENCY         (1),                //positive integer;
  //.FIFO_READ_LATENCY         (3),                //positive integer;
  .READ_DATA_WIDTH           (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .RD_DATA_COUNT_WIDTH       ($clog2(LP_WR_FIFO_DEPTH)),               //positive integer, not used
  .PROG_EMPTY_THRESH         (10),               //positive integer, not used 
  .DOUT_RESET_VALUE          ("0"),              //string, don't care
  .WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

) inst_wr_xpm_fifo_sync (
  .sleep         ( 1'b0                    ) ,
  .rst           ( areset           ) ,
  //.rst           ( areset | ap_start_pulse | ap_done_r ),
  .wr_clk        ( ap_clk                  ) ,
  .wr_en         ( adder_tvalid            ) ,
  .din           ( adder_tdata             ) ,
  //.full          ( adder_tready_n          ) ,
  .full          ( adder_tready_n          ) ,
  .prog_full     (                         ) ,
  .wr_data_count (                         ) ,
  .overflow      (                         ) ,
  .wr_rst_busy   (                         ) ,
  .rd_en         ( wr_fifo_tready          ) ,
  .dout          ( wr_fifo_tdata           ) ,
  .empty         ( wr_fifo_tvalid_n        ) ,
  .prog_empty    (                         ) ,
  .rd_data_count (                         ) ,
  .underflow     (                         ) ,
  .rd_rst_busy   (                         ) ,
  .injectsbiterr ( 1'b0                    ) ,
  .injectdbiterr ( 1'b0                    ) ,
  .sbiterr       (                         ) ,
  .dbiterr       (                         ) 

);

assign finished_r = finished;

/*
// AXI4 Write Master
krnl_vadd_rtl_axi_write_master #( 
  .C_ADDR_WIDTH       ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
  .C_DATA_WIDTH       ( C_M_AXI_GMEM_DATA_WIDTH ) ,
  .C_MAX_LENGTH_WIDTH ( LP_LENGTH_WIDTH         ) ,
  .C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
  .C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) 
)
inst_axi_write_master ( 
  .aclk        ( ap_clk             ) ,
  .areset      ( areset             ) ,

  .ctrl_start  ( ap_start_pulse     ) ,
  //.ctrl_start  ( out_pulse          ) ,
  .ctrl_offset ( c + 16'h1000       ) ,
  .ctrl_length ( length_r           ) ,
  //.ctrl_length ( num_out_tx         ) ,
  //.ctrl_length ( 12'd3200            ) , // This amounts to exactly 50 transactions
  //.ctrl_length ( 13'd4096            ) , 
  
  
  // Result: Success if `m_axis_tvalid` is asserted while
  // `axi_io_helper` is in `s_idle` state
  //.ctrl_length ( 32'd4096            ) , 

  // Result: Success if `finished_r` influences `s_tvalid`
  //.ctrl_length ( 32'd64            ) ,

  // Result: Success even when `finished_r` is not
  // influencing `s_tvalid`
  //.ctrl_length ( 32'd50            ) ,

  //.ctrl_length ( 32'd8192            ) , // Result: TODO
  //.ctrl_length ( 32'd3200            ) , // This amounts to exactly 50 transactions
  //.ctrl_length ( num_out_tx << 6        ) ,
  //.ctrl_done   ( write_done         ) ,
  .ctrl_done   ( ap_done         ) ,

  .awvalid     ( m_axi_gmem_AWVALID ) ,
  .awready     ( m_axi_gmem_AWREADY ) ,
  .awaddr      ( m_axi_gmem_AWADDR  ) ,
  .awlen       ( m_axi_gmem_AWLEN   ) ,
  .awsize      ( m_axi_gmem_AWSIZE  ) ,

  .s_tvalid    ( ~wr_fifo_tvalid_n ) ,
  //.s_tvalid    ( ~wr_fifo_tvalid_n | finished_r  ) ,
  .s_tready    ( wr_fifo_tready     ) ,
  .s_tdata     ( wr_fifo_tdata      ) ,

  .wvalid      ( m_axi_gmem_WVALID  ) ,
  .wready      ( m_axi_gmem_WREADY  ) ,
  .wdata       ( m_axi_gmem_WDATA   ) ,
  .wstrb       ( m_axi_gmem_WSTRB   ) ,
  .wlast       ( m_axi_gmem_WLAST   ) ,

  .bvalid      ( m_axi_gmem_BVALID  ) ,
  .bready      ( m_axi_gmem_BREADY  ) ,
  .bresp       ( m_axi_gmem_BRESP   ) 
);
*/

// AXI4 Write Master
alt_write_axi_master #(
  .C_M_AXI_ADDR_WIDTH  ( C_M_AXI_GMEM_ADDR_WIDTH    ) ,
  .C_M_AXI_DATA_WIDTH  ( C_M_AXI_GMEM_DATA_WIDTH    ) ,
  .C_XFER_SIZE_WIDTH   ( LP_LENGTH_WIDTH            ) ,
  .C_INCLUDE_DATA_FIFO ( 1                          )
)
inst_axi_write_master (
  .aclk                    ( ap_clk                 ) ,
  .areset                  ( areset                 ) ,

  //.ctrl_start              ( ap_start_pulse         ) ,
  .ctrl_start              ( out_pulse              ) ,
  .ctrl_done               ( ap_done                ) ,
  .ctrl_addr_offset        ( c_reg                  ) ,
  //.ctrl_xfer_size_in_bytes ( length_r << 6          ) ,
  //.ctrl_xfer_size_in_bytes ( length_r_reg << 6      ) ,
  .ctrl_xfer_size_in_bytes ( num_out_tx << 6        ) ,

  .m_axi_awvalid           ( m_axi_gmem_AWVALID     ) ,
  .m_axi_awready           ( m_axi_gmem_AWREADY     ) ,
  .m_axi_awaddr            ( m_axi_gmem_AWADDR      ) ,
  .m_axi_awlen             ( m_axi_gmem_AWLEN       ) ,

  .m_axi_wvalid            ( m_axi_gmem_WVALID      ) ,
  .m_axi_wready            ( m_axi_gmem_WREADY      ) ,
  .m_axi_wdata             ( m_axi_gmem_WDATA       ) ,
  .m_axi_wstrb             ( m_axi_gmem_WSTRB       ) ,
  .m_axi_wlast             ( m_axi_gmem_WLAST       ) ,

  .m_axi_bvalid            ( m_axi_gmem_BVALID      ) ,
  .m_axi_bready            ( m_axi_gmem_BREADY      ) ,

  .s_axis_aclk             ( ap_clk                 ) ,
  .s_axis_areset           ( areset                 ) ,

  .s_axis_tvalid           ( ~wr_fifo_tvalid_n      ) ,
  .s_axis_tready           ( wr_fifo_tready         ) ,
  .s_axis_tdata            ( wr_fifo_tdata          ) ,

  .awsize                  ( m_axi_gmem_AWSIZE      )
);

endmodule : krnl_vadd_rtl_int

`default_nettype wire
