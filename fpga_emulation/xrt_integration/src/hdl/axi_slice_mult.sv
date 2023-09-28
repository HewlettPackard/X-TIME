// The following implements a chain of `NUM_SLICES`
// fully-registered register slices.
// We use it to improve timing closure.
module slice_mult
#(
    parameter NUM_SLICES = 1
) (
    input  logic                    clk,
    input  logic                    rst,

    input  logic                    us_valid,
    input  logic [512-1:0]        us_data,
    output logic                    us_ready,

    output logic                    ds_valid,
    output logic [512-1:0]        ds_data,
    input  logic                    ds_ready
);
    logic [NUM_SLICES:0][512-1:0] tdata_inter;
    logic [NUM_SLICES:0]            tvalid_inter, tready_inter;

    assign tvalid_inter[0] = us_valid;
    assign tdata_inter[0] = us_data;
    assign us_ready = tready_inter[0];

    assign ds_valid = tvalid_inter[NUM_SLICES];
    assign ds_data = tdata_inter[NUM_SLICES];
    assign tready_inter[NUM_SLICES] = ds_ready;

    genvar i;
    generate
        for (i = 0; i < NUM_SLICES; i++)
        begin: register_slice_chain
            axis_register_slice_0 i_register_slice
            (
                .aclk (clk),
                .aresetn (~rst),

                .s_axis_tready(tready_inter[i]),
                .s_axis_tvalid(tvalid_inter[i]),
                .s_axis_tdata(tdata_inter[i]),

                .m_axis_tready(tready_inter[i+1]),
                .m_axis_tvalid(tvalid_inter[i+1]),
                .m_axis_tdata(tdata_inter[i+1])
            );
        end
    endgenerate
endmodule
