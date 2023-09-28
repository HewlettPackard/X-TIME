// Ref: https://www.itdev.co.uk/blog/pipelining-axi-buses-registered-ready-signals
module pipe_adapter
#(
    parameter WIDTH = 512
) (
    input  logic                clk,
    input  logic                rst,

    input  logic                us_valid,
    input  logic [WIDTH-1:0]    us_data,
    output logic                us_ready,

    output logic                ds_valid,
    output logic [WIDTH-1:0]    ds_data,
    input  logic                ds_ready
);
    reg [WIDTH-1:0]             expansion_data_reg  = 0;
    reg                         expansion_valid_reg = 0;

    reg [WIDTH-1:0]             primary_data_reg    = 0;
    reg                         primary_valid_reg   = 0;

    always_ff @(posedge clk) begin
        if (rst) begin
            primary_valid_reg   <= 0;
            expansion_valid_reg <= 0;
            primary_data_reg    <= 0;
            expansion_data_reg  <= 0;
        end else begin
            // Accept data if ready is high
            if (us_ready) begin
                primary_valid_reg   <= us_valid;
                primary_data_reg    <= us_data;

                // When ds is not ready, accept data into expansion reg until it is valid
                if (!ds_ready) begin
                    expansion_valid_reg <= primary_valid_reg;
                    expansion_data_reg  <= primary_data_reg;
                end
            end

            // When ds becomes ready the expansion reg data is accepted and we must clear the valid register
            if (ds_ready) begin
                expansion_valid_reg <= 0;
            end
        end
    end

    // Ready as long as there is nothing in the expansion register
    assign us_ready = ~expansion_valid_reg;

    // Selecting the expansion register if it has valid data
    assign ds_valid = expansion_valid_reg || primary_valid_reg;
    assign ds_data  = expansion_valid_reg ? expansion_data_reg : primary_data_reg;

endmodule
