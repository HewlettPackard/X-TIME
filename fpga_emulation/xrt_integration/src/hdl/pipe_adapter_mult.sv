// The following implements a chain of `NUM_SLICES`
// fully-registered register slices.
// We use it to improve timing closure.
module pipe_adapter_mult
#(
    parameter WIDTH = 512,
    parameter NUM_SLICES = 1
) (
    input  logic                    clk,
    input  logic                    rst,

    input  logic                    us_valid,
    input  logic [WIDTH-1:0]        us_data,
    output logic                    us_ready,

    output logic                    ds_valid,
    output logic [WIDTH-1:0]        ds_data,
    input  logic                    ds_ready
);
    logic [NUM_SLICES:0][WIDTH-1:0] tdata_inter;
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
        begin: pipe_adapter_chain
            pipe_adapter
            #(
                .WIDTH(WIDTH)
            )
            i_pipe_adapter
            (
                .clk(clk),
                .rst(rst),

                .us_valid(tvalid_inter[i]),
                .us_ready(tready_inter[i]),
                .us_data(tdata_inter[i]),

                .ds_valid(tvalid_inter[i+1]),
                .ds_ready(tready_inter[i+1]),
                .ds_data(tdata_inter[i+1])
            );
        end
    endgenerate
endmodule
