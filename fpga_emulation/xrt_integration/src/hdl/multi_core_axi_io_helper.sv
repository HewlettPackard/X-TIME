`include "all_params.sv"

module multi_core_axi_io_helper
#(
    parameter NUM_MODE_CHANNELS = NUM_CORES,
    parameter NUM_ROUTER_OUTPUTS = 1,
    parameter integer NUM_CORES_ID_WIDTH = int'($ceil($clog2(NUM_CORES)))
)
(
    input wire                                      clk,
    input wire                                      rst,

    input logic                                     s_axis_tvalid,
    output logic                                    s_axis_tready,
    input logic  [C_AXIS_TDATA_WIDTH-1:0]           s_axis_tdata,
    //input logic  [C_AXIS_TDATA_WIDTH/8-1:0]        s_axis_tkeep,
    //input logic                                    s_axis_tlast,

    output logic                                    m_axis_tvalid,
    input wire                                      m_axis_tready,
    output logic [C_AXIS_TDATA_WIDTH-1:0]           m_axis_tdata,

    output logic [$clog2(NUM_CLAUSES)-1:0]          clause,
    output logic [$clog2(NUM_VARS)-1:0]             variable,
    output logic                                    threshold_kind,
    output logic [2:0]                              mode [NUM_MODE_CHANNELS-1:0],
    input logic                                     core_s_axis_ready,
    output logic [VAR_WIDTH-1:0]                    input_vars [NUM_VARS-1:0],
    input logic                                     ml_valid,
    output logic                                    ml_ready,
    input wire   [NUM_CLAUSES-1:0]                  match_lines,
    output logic signed [LEAF_VALUES_NUM_BITS-1:0]  leaf_value,
    output logic [TREE_ID_NUM_BITS-1:0]             tree_id,
    output logic [CLASS_ID_NUM_BITS-1:0]            class_id,
    input logic  signed [LEAF_VALUES_NUM_BITS-1:0]  match_leaves [NUM_ROUTER_OUTPUTS-1:0],
    input logic  [TREE_ID_NUM_BITS-1:0]             match_tree_ids [NUM_ROUTER_OUTPUTS-1:0],
    input logic  [CLASS_ID_NUM_BITS-1:0]            match_class_ids [NUM_ROUTER_OUTPUTS-1:0],
    output logic [C_MAX_LENGTH_WIDTH-1:0]           num_out_tx,
    output logic                                    out_pulse,
    output logic                                    finished,
    output logic                                    trig_out,
    input logic                                     trig_out_ack
);

localparam integer CLAUSE_ID_WIDTH = $clog2(NUM_CLAUSES + 1);
localparam integer TIGHT_CLAUSE_ID_WIDTH = $clog2(NUM_CLAUSES);
localparam integer EXPANDED_VAR_ID_WIDTH = $clog2(NUM_VARS + 1);
localparam integer TIGHT_VAR_ID_WIDTH = $clog2(NUM_VARS);
localparam integer MAX_MULT = C_AXIS_TDATA_WIDTH / MACRO_VAR_WIDTH - 1;
localparam integer MAX_FINE_MULT = C_AXIS_TDATA_WIDTH / VAR_WIDTH - 1;
localparam integer SINGLE_MATCH_BITS = LEAF_VALUES_NUM_BITS + TREE_ID_NUM_BITS + CLASS_ID_NUM_BITS;
localparam integer TOTAL_MATCH_BITS = NUM_ROUTER_OUTPUTS * SINGLE_MATCH_BITS;

reg   [EXPANDED_VAR_ID_WIDTH-1:0]               num_vars_per_core;
reg   [CLAUSE_ID_WIDTH-1:0]                     num_clauses_per_core;
reg   [EXPANDED_VAR_ID_WIDTH-1:0]               num_dc_per_core;
reg   [QUERY_ID_WIDTH-1:0]                      num_queries_per_core;

reg   [CLAUSE_ID_WIDTH-1:0]                     current_clause;
reg   [EXPANDED_VAR_ID_WIDTH-1:0]               current_dont_care;
reg   [QUERY_ID_WIDTH-1:0]                      current_query;
reg                                             current_tkind;

reg   [NUM_CORES_ID_WIDTH-1:0]                  current_core;
reg   [NUM_CORES_ID_WIDTH-1:0]                  max_core_id;

reg   [C_MAX_LENGTH_WIDTH-1:0]                  num_out_tx_r;

reg   [31:0]                                    debug_num_completed_inputs;
reg   [31:0]                                    debug_num_completed_inputs_through_reg;
reg   [31:0]                                    debug_num_completed_ml_txs;

always @(posedge clk)
begin
    if (s_axis_tvalid && s_axis_tready)
    begin
        debug_num_completed_inputs <= debug_num_completed_inputs + 1;
    end

    if (ml_valid && ml_ready)
    begin
        debug_num_completed_ml_txs <= debug_num_completed_ml_txs + 1;
    end

    if (s_axis_tvalid_aux && s_axis_tready_aux)
    begin
        debug_num_completed_inputs_through_reg <= debug_num_completed_inputs_through_reg + 1;
    end
end

enum int unsigned { s_start_pulse = 0, s_read_param = 2, s_conf_thresholds = 4, s_conf_dc = 8, s_query = 16, s_idle = 32} state = s_read_param;

logic core_s_axis_valid;
logic m_axis_tvalid_c;
reg m_axis_tvalid_r = 0;
reg active_leaf_values = 0;

logic  [C_AXIS_TDATA_WIDTH-1:0]                 final_match_data;

logic                                           s_axis_tvalid_aux;
logic                                           s_axis_tready_aux;
logic  [C_AXIS_TDATA_WIDTH-1:0]                 s_axis_tdata_aux;

always_comb
begin
    num_out_tx = num_out_tx_r;

    s_axis_tready_aux = 0;
    finished = 0;
    m_axis_tvalid_c = 0;

    // This assumes that whenever any core is consuming
    // query inputs, mode[0] reflects that behavior.
    core_s_axis_valid = (mode[0] == 0) || (mode[0] == 4);

    // TODO: ENSURE A LATCH IS NOT INFERRED FOR
    //       final_match_data
    // See:  _x.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/logs/link/syn/ulp_krnl_vadd_rtl_1_0_synth_1_runme.log
    //       Line 406
    if (active_leaf_values)
    begin
        if (TOTAL_MATCH_BITS < C_AXIS_TDATA_WIDTH)
        begin
            final_match_data[C_AXIS_TDATA_WIDTH-1:TOTAL_MATCH_BITS] = 0;
        end

        for (int i=0; i < NUM_ROUTER_OUTPUTS && ((i + 1) * SINGLE_MATCH_BITS <= C_AXIS_TDATA_WIDTH); i++)
        begin
            int leaf_value_start = (LEAF_VALUES_NUM_BITS + TREE_ID_NUM_BITS + CLASS_ID_NUM_BITS) * i;
            int tree_id_start = leaf_value_start + LEAF_VALUES_NUM_BITS;
            int class_id_start = tree_id_start + TREE_ID_NUM_BITS;

            final_match_data[leaf_value_start+:LEAF_VALUES_NUM_BITS] <= match_leaves[i];
            final_match_data[tree_id_start+:TREE_ID_NUM_BITS] <= match_tree_ids[i];
            final_match_data[class_id_start+:CLASS_ID_NUM_BITS] <= match_class_ids[i];
        end
    end else begin
        final_match_data = {{C_AXIS_TDATA_WIDTH-NUM_CLAUSES{1'b0}}, match_lines};
    end

    if (!rst)
    begin
        case(state)
            s_read_param:
            begin
                s_axis_tready_aux = 1;
            end
            s_start_pulse:
            begin
            end
            s_conf_thresholds:
            begin
                s_axis_tready_aux = 1;
            end
            s_conf_dc:
            begin
                s_axis_tready_aux = num_dc_per_core == current_dont_care + 1;
            end
            s_query:
            begin
                s_axis_tready_aux = core_s_axis_ready || !core_s_axis_valid;
            end
            s_idle:
            begin
                finished = 1;
                m_axis_tvalid_c = 1;
            end
            default:
            begin
            end
        endcase
    end

    //ml_ready = m_axis_tready || !m_axis_tvalid;
    //m_axis_tvalid = m_axis_tvalid_c || m_axis_tvalid_r;
end

always_ff @(posedge clk)
begin
    out_pulse <= 0;

    for (int i = 0; i < NUM_MODE_CHANNELS; i++)
    begin
        mode[i] <= 3'd3;
    end

    if(rst) begin
        state <= s_read_param;
        current_tkind <= 0;
        current_clause <= 0;
        current_dont_care <= 0;
        current_query <= 0;
        current_core <= 0;

        num_vars_per_core <= 0;
        num_clauses_per_core <= 0;
        num_dc_per_core <= 0;
        num_queries_per_core <= 0;
        max_core_id <= 0;

        debug_num_completed_inputs <= 0;
        debug_num_completed_inputs_through_reg <= 0;
        debug_num_completed_ml_txs <= 0;
    end
    else begin
        case(state)
            s_read_param:
            begin
                if (s_axis_tvalid_aux)
                begin
                    // TODO: Parametrize the 32-bit offset steps
                    num_clauses_per_core <= s_axis_tdata_aux[0+:CLAUSE_ID_WIDTH];
                    num_dc_per_core <= s_axis_tdata_aux[32+:EXPANDED_VAR_ID_WIDTH];
                    num_queries_per_core <= s_axis_tdata_aux[64+:QUERY_ID_WIDTH];
                    num_vars_per_core <= s_axis_tdata_aux[96+:EXPANDED_VAR_ID_WIDTH];
                    active_leaf_values <= s_axis_tdata_aux[128+:1];
                    max_core_id <= s_axis_tdata_aux[160+:(NUM_CORES_ID_WIDTH+1)] - 1;
                    num_out_tx_r <= s_axis_tdata_aux[64+:QUERY_ID_WIDTH];

                    state <= s_start_pulse;
                end
            end
            s_start_pulse:
            begin
                    out_pulse <= 1;
                    state <= s_conf_thresholds;
            end
            s_conf_thresholds:
            begin
                if (s_axis_tvalid_aux)
                begin
                    if (current_tkind)
                    begin
                        current_clause <= current_clause + 1;
                        leaf_value <= s_axis_tdata_aux[MAX_MULT*MACRO_VAR_WIDTH+:LEAF_VALUES_NUM_BITS];
                    end else begin
                        tree_id <= s_axis_tdata_aux[MAX_MULT*MACRO_VAR_WIDTH+:TREE_ID_NUM_BITS];
                        class_id <= s_axis_tdata_aux[(MAX_MULT*MACRO_VAR_WIDTH+TREE_ID_NUM_BITS)+:CLASS_ID_NUM_BITS];
                    end

                    current_tkind <= !current_tkind;

                    mode[current_core] <= 3'd1;
                    clause <= current_clause[TIGHT_CLAUSE_ID_WIDTH-1:0];
                    threshold_kind <= current_tkind;

                    if ((num_clauses_per_core == current_clause + 1) && (current_tkind == 1))                
                    begin
                        if (current_core == max_core_id)
                        begin
                            current_core <= 0;
                            if (num_dc_per_core > 0)
                            begin
                                state <= s_conf_dc;
                            end else begin
                                state <= s_query;
                            end
                        end else begin
                            current_core <= current_core + 1;
                            current_tkind <= 0;
                            current_clause <= 0;
                        end
                    end
                    
                    // TODO: Remove the requirement that NUM_VARS * VAR_WIDTH <= C_AXIS_TDATA_WIDTH
                    for (int i=0; i < NUM_VARS && i <= MAX_FINE_MULT; i++)
                    begin
                        if (i < num_vars_per_core)
                        begin
                            input_vars[i] <= s_axis_tdata_aux[i*VAR_WIDTH+:VAR_WIDTH];
                        end else begin
                            if (current_tkind)
                            begin
                                input_vars[i] <= MAX_STATE;
                            end else begin
                                input_vars[i] <= 0;
                            end
                        end
                    end

                    for (int i=(MAX_FINE_MULT + 1); i < NUM_VARS; i++)
                    begin
                        if (current_tkind)
                        begin
                            input_vars[i] <= MAX_STATE;
                        end else begin
                            input_vars[i] <= 0;
                        end
                    end
                end
            end
            s_conf_dc:
            begin
                if (s_axis_tvalid_aux)
                begin
                    mode[current_core] <= 3'd2;
                    // TODO: Avoid using multiplication here
                    //       by iteratively adding to an address
                    //       register.
                    variable <= s_axis_tdata_aux[current_dont_care*MACRO_VAR_WIDTH+:TIGHT_VAR_ID_WIDTH];

                    current_dont_care <= current_dont_care + 1;

                    if (num_dc_per_core == current_dont_care + 1)                
                    begin
                        if (current_core == max_core_id)
                        begin
                            current_core <= 0;
                            state <= s_query;
                        end else begin
                            current_core <= current_core + 1;
                            current_dont_care <= 0;
                        end
                    end
                end
            end
            s_query:
            begin

                if (m_axis_tready && m_axis_tvalid)
                begin
                    if (num_queries_per_core == current_query + 1)                
                    begin
                        current_query <= 0;
                        state <= s_idle;
                    end else begin
                        current_query <= current_query + 1;
                    end
                end else begin
                    current_query <= current_query;
                end

                if (s_axis_tready_aux)
                begin
                    // TODO: Remove the requirement that NUM_VARS * VAR_WIDTH <= C_AXIS_TDATA_WIDTH
                    for (int i=0; i < NUM_VARS && i <= MAX_FINE_MULT; i++)
                    begin
                        if (i < num_vars_per_core)
                        begin
                            input_vars[i] <= s_axis_tdata_aux[i*VAR_WIDTH+:VAR_WIDTH];
                        end else begin
                            input_vars[i] <= 0;
                        end
                    end

                    for (int i=(MAX_FINE_MULT + 1); i < NUM_VARS; i++)
                    begin
                        input_vars[i] <= 0;
                    end

                    if (s_axis_tvalid_aux)
                    begin
                        if (active_leaf_values)
                        begin
                            for (int i = 0; i < NUM_MODE_CHANNELS; i++)
                            begin
                                mode[i] <= 3'd4;
                            end
                        end else begin
                            for (int i = 0; i < NUM_MODE_CHANNELS; i++)
                            begin
                                mode[i] <= 3'd0;
                            end
                        end
                    end else begin
                        for (int i = 0; i < NUM_MODE_CHANNELS; i++)
                        begin
                            mode[i] <= 3'd3;
                        end
                    end
                end else begin
                    for (int i = 0; i < NUM_MODE_CHANNELS; i++)
                    begin
                        mode[i] <= mode[i];
                    end
                end
            end
            s_idle:
            begin
                //m_axis_tdata[0+:32] <= 32'h1234_aabb;
                //m_axis_tvalid_r <= 1;
            end
            default:
            begin
            end
        endcase

    end
end

slice_mult #( 
  .NUM_SLICES   ( 3 )
)
i_input_slice_mult ( 
  .clk          ( clk                                                   ) ,
  .rst          ( rst                                                   ) ,

  //.us_valid     ( ml_valid || finished                                  ) ,
  .us_valid     ( ml_valid                                              ) ,
  .us_data      ( final_match_data                                      ) ,
  .us_ready     ( ml_ready                                              ) ,

  .ds_valid     ( m_axis_tvalid                                         ) ,
  .ds_data      ( m_axis_tdata                                          ) ,
  .ds_ready     ( m_axis_tready                                         ) 
);

slice_mult #( 
  .NUM_SLICES   ( 3 )
)
i_output_slice_mult ( 
  .clk          ( clk                                                   ) ,
  .rst          ( rst                                                   ) ,

  .us_valid     ( s_axis_tvalid                                         ) ,
  .us_data      ( s_axis_tdata                                          ) ,
  .us_ready     ( s_axis_tready                                         ) ,

  .ds_valid     ( s_axis_tvalid_aux                                     ) ,
  .ds_data      ( s_axis_tdata_aux                                      ) ,
  .ds_ready     ( s_axis_tready_aux                                     )
);

/*
ila_1 i_ila_1 (
    .clk                   ( clk                   ) ,
    .probe0                ( num_vars_per_core        ) ,
    .probe1                ( num_clauses_per_core     ) ,
    .probe2                ( num_dc_per_core          ) ,
    .probe3                ( num_queries_per_core     ) ,
    .probe4                ( current_clause        ) ,
    .probe5                ( next_clause           ) ,
    .probe6                ( current_dont_care     ) ,
    .probe7                ( next_dont_care        ) ,
    .probe8                ( current_query         ) ,
    .probe9                ( next_query            ) ,
    .probe10               ( current_tkind         ) ,
    .probe11               ( next_tkind            ) ,
    .probe12               ( state                 ) ,
    .probe13               ( next_state            ) ,
    .probe14               ( clause                ) ,
    .probe15               ( variable              ) ,
    .probe16               ( threshold_kind        ) ,
    .probe17               ( mode[0]               ) ,
    .probe18               ( match_lines[15:0]     ) ,
    .probe19               ( finished              ) ,
    .probe20               ( s_axis_tvalid_aux     ) ,
    .probe21               ( s_axis_tready_aux     ) ,
    .probe22               ( s_axis_tdata_aux      ) ,
    .trig_out              ( trig_out              ) ,
    .trig_out_ack          ( trig_out_ack          ) ,
    .probe23               ( rst                   )
);
*/

endmodule
