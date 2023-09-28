`include "all_params.sv"

module cam_core
#(
    parameter NUM_OUTPUTS = MAX_MATCHES,
    parameter MAX_FLATTENED_WIDTH = 512,
    parameter NUM_INPUT_STAGES = NUM_CORE_INPUT_STAGES
)
(
    input wire                                      clk,
    input wire                                      rst,

    input logic [$clog2(NUM_CLAUSES)-1:0]           clause,

    // This is only used for setting variable don't cares.
    // That operation must be done after setting all lower and higher thresholds,
    // since it is fulfilled by updating the accepted range for that variable
    // such that any of its possible values is a match.
    input logic [$clog2(NUM_VARS)-1:0]              variable,

    // During threshold setting, this indicates whether we are modifying lower
    // or higher thresholds.
    input wire                                      threshold_kind,

    // Determines whether we are doing inference, modifying the clause thresholds, setting variable don't cares
    input logic [2:0]                               mode,
    // Set of variable values to be queried against the clauses
    input wire [VAR_WIDTH-1:0]                      input_vars [NUM_VARS-1:0],
    input logic signed [LEAF_VALUES_NUM_BITS-1:0]   input_leaf_value,
    input logic [TREE_ID_NUM_BITS-1:0]              tree_id,
    input logic [CLASS_ID_NUM_BITS-1:0]             class_id,

    output logic                                    core_s_axis_ready,

    // Concatenation of all match lines
    output logic [NUM_CLAUSES-1:0]                  match_lines,
    output logic signed [LEAF_VALUES_NUM_BITS-1:0]  match_leaves [NUM_OUTPUTS-1:0],
    output logic [TREE_ID_NUM_BITS-1:0]             match_tree_ids [NUM_OUTPUTS-1:0],
    output logic [CLASS_ID_NUM_BITS-1:0]            match_class_ids [NUM_OUTPUTS-1:0],
    input logic                                     ml_ready,
    output logic                                    ml_valid
);

localparam                                          NUM_INPUT_REPLICAS = NUM_CLAUSES / NUM_REPLICA_CONSUMERS;

reg [VAR_WIDTH-1:0]                                 input_vars_aux_replicas [NUM_INPUT_REPLICAS-1:0][NUM_VARS-1:0];

reg [VAR_WIDTH-1:0]                                 lower_thresholds  [NUM_CLAUSES-1:0][NUM_VARS-1:0];
reg [VAR_WIDTH-1:0]                                 higher_thresholds [NUM_CLAUSES-1:0][NUM_VARS-1:0];
reg signed [LEAF_VALUES_NUM_BITS-1:0]               leaf_values [NUM_CLAUSES-1:0];
reg [TREE_ID_NUM_BITS-1:0]                          tree_ids [NUM_CLAUSES-1:0];
reg [CLASS_ID_NUM_BITS-1:0]                         class_ids [NUM_CLAUSES-1:0];

reg rst_1;
reg rst_2;
reg rst_3;

reg [NUM_VARS-1:0]                                  match_bits_dont_care_replicas [NUM_INPUT_REPLICAS-1:0];
reg [NUM_CLAUSES-1:0]                               enabled_clauses;

logic [NUM_VARS-1:0]                                match_bits [NUM_CLAUSES-1:0];
logic [NUM_VARS-1:0]                                match_bits_aux [NUM_CLAUSES-1:0];
logic [NUM_VARS-1:0]                                p2_match_bits_aux [NUM_CLAUSES-1:0];
logic [NUM_REPLICA_CONSUMERS*NUM_VARS-1:0]          flattened_match_bits_aux [NUM_INPUT_REPLICAS-1:0];
logic [NUM_REPLICA_CONSUMERS*NUM_VARS-1:0]          p2_flattened_match_bits_aux [NUM_INPUT_REPLICAS-1:0];
logic [NUM_CLAUSES-1:0]                             match_lines_aux;

// TODO: Ensure all these variables are properly
//       initialized during a reset
logic [2:0]                                         mode_aux;
logic                                               mode_valid_aux;
logic                                               core_s_axis_ready_aux;
logic                                               passthrough;

wire [NUM_VARS * VAR_WIDTH-1:0]                     flattened_input_vars;
wire [NUM_VARS * VAR_WIDTH-1:0]                     flattened_input_vars_aux;
wire [VAR_WIDTH-1:0]                                input_vars_aux [NUM_VARS-1:0];

logic [$clog2(NUM_CLAUSES)-1:0]                     clause_aux;
logic [$clog2(NUM_VARS)-1:0]                        variable_aux;
logic                                               threshold_kind_aux;
logic signed [LEAF_VALUES_NUM_BITS-1:0]             input_leaf_value_aux;
logic [TREE_ID_NUM_BITS-1:0]                        tree_id_aux;
logic [CLASS_ID_NUM_BITS-1:0]                       class_id_aux;

logic                                   p0_ready;
logic                                   p0_valid;
logic [2:0]                             p0_mode;
logic                                   p0_tkind;
logic signed [LEAF_VALUES_NUM_BITS-1:0] p0_input_leaf_value_aux;
logic [TREE_ID_NUM_BITS-1:0]            p0_tree_id_aux;
logic [CLASS_ID_NUM_BITS-1:0]           p0_class_id_aux;
logic [$clog2(NUM_CLAUSES)-1:0]         p0_clause_aux;

//logic                                   p1_ready;
logic [NUM_INPUT_REPLICAS-1:0]          p1_ready_replicas;
logic                                   p1_valid;
logic signed [LEAF_VALUES_NUM_BITS-1:0] p1_input_leaf_value_aux;
logic [TREE_ID_NUM_BITS-1:0]            p1_tree_id_aux;
logic [CLASS_ID_NUM_BITS-1:0]           p1_class_id_aux;

logic [2:0]                             p1_mode_replicas [NUM_INPUT_REPLICAS-1:0];
logic                                   p1_tkind_replicas [NUM_INPUT_REPLICAS-1:0];
logic [$clog2(NUM_CLAUSES)-1:0]         p1_clause_aux_replicas [NUM_INPUT_REPLICAS-1:0];

logic       p2_ready;
logic       p2_valid;

logic       p3_ready;
logic       p3_valid;

logic       p4_ready;
logic       p4_valid;

logic       p5_ready;
logic       p5_valid;

generate
    genvar i;

    for (i = 0; i < NUM_VARS; i++)
    begin
        // These transformations are used by `i_cc_slice_mult_1`
        assign flattened_input_vars[i * VAR_WIDTH +: VAR_WIDTH] = input_vars[i];
        assign input_vars_aux[i] = flattened_input_vars_aux[i * VAR_WIDTH +: VAR_WIDTH];
    end
endgenerate

//////////////////////////////////////////////////////////////
// Register slices directly connected to `cam_core` interfaces
//////////////////////////////////////////////////////////////

slice_mult #( 
  .NUM_SLICES   ( NUM_INPUT_STAGES )
)
i_cc_slice_mult_0 ( 
  .clk          ( clk                                                   ) ,
  .rst          ( rst_3                                                 ) ,

  .us_valid     ( mode != 3                                             ) ,
  .us_data      ( mode                                                  ) ,
  .us_ready     ( core_s_axis_ready                                     ) ,

  .ds_valid     ( mode_valid_aux                                        ) ,
  .ds_data      ( mode_aux                                              ) ,
  .ds_ready     ( core_s_axis_ready_aux || passthrough                  )
);

slice_mult #( 
  .NUM_SLICES   ( NUM_INPUT_STAGES )
)
i_cc_slice_mult_1 ( 
  .clk          ( clk                                                   ) ,
  .rst          ( rst_3                                                 ) ,

  .us_valid     ( mode != 3                                             ) ,
  .us_data      ( flattened_input_vars                                  ) ,
  .us_ready     (                                                       ) ,

  // We do not need to handle the valid here because
  // it should be identical to mode_valid_aux
  .ds_valid     (                                                       ) ,
  .ds_data      ( flattened_input_vars_aux                              ) ,
  .ds_ready     ( core_s_axis_ready_aux || passthrough                  )
);

slice_mult #( 
  .NUM_SLICES   ( NUM_INPUT_STAGES )
)
i_cc_slice_mult_2 ( 
  .clk          ( clk                                                   ) ,
  .rst          ( rst_3                                                 ) ,

  .us_valid     ( mode != 3                                             ) ,
  .us_data      ( input_leaf_value                                      ) ,
  .us_ready     (                                                       ) ,

  // We do not need to handle the valid here because
  // it should be identical to mode_valid_aux
  .ds_valid     (                                                       ) ,
  .ds_data      ( input_leaf_value_aux                                  ) ,
  .ds_ready     ( core_s_axis_ready_aux || passthrough                  )
);

slice_mult #( 
  .NUM_SLICES   ( NUM_INPUT_STAGES )
)
i_cc_slice_mult_3 ( 
  .clk          ( clk                                                   ) ,
  .rst          ( rst_3                                                 ) ,

  .us_valid     ( mode != 3                                             ) ,
  .us_data      ( tree_id                                               ) ,
  .us_ready     (                                                       ) ,

  // We do not need to handle the valid here because
  // it should be identical to mode_valid_aux
  .ds_valid     (                                                       ) ,
  .ds_data      ( tree_id_aux                                           ) ,
  .ds_ready     ( core_s_axis_ready_aux || passthrough                  )
);

slice_mult #( 
  .NUM_SLICES   ( NUM_INPUT_STAGES )
)
i_cc_slice_mult_4 ( 
  .clk          ( clk                                                   ) ,
  .rst          ( rst_3                                                 ) ,

  .us_valid     ( mode != 3                                             ) ,
  .us_data      ( class_id                                              ) ,
  .us_ready     (                                                       ) ,

  // We do not need to handle the valid here because
  // it should be identical to mode_valid_aux
  .ds_valid     (                                                       ) ,
  .ds_data      ( class_id_aux                                          ) ,
  .ds_ready     ( core_s_axis_ready_aux || passthrough                  )
);

slice_mult #( 
  .NUM_SLICES   ( NUM_INPUT_STAGES )
)
i_cc_slice_mult_5 ( 
  .clk          ( clk                                                   ) ,
  .rst          ( rst_3                                                 ) ,

  .us_valid     ( mode != 3                                             ) ,
  .us_data      ( clause                                                ) ,
  .us_ready     (                                                       ) ,

  // We do not need to handle the valid here because
  // it should be identical to mode_valid_aux
  .ds_valid     (                                                       ) ,
  .ds_data      ( clause_aux                                            ) ,
  .ds_ready     ( core_s_axis_ready_aux || passthrough                  )
);

slice_mult #( 
  .NUM_SLICES   ( NUM_INPUT_STAGES )
)
i_cc_slice_mult_6 ( 
  .clk          ( clk                                                   ) ,
  .rst          ( rst_3                                                 ) ,

  .us_valid     ( mode != 3                                             ) ,
  .us_data      ( variable                                              ) ,
  .us_ready     (                                                       ) ,

  // We do not need to handle the valid here because
  // it should be identical to mode_valid_aux
  .ds_valid     (                                                       ) ,
  .ds_data      ( variable_aux                                          ) ,
  .ds_ready     ( core_s_axis_ready_aux || passthrough                  )
);

slice_mult #( 
  .NUM_SLICES   ( NUM_INPUT_STAGES )
)
i_cc_slice_mult_7 ( 
  .clk          ( clk                                                   ) ,
  .rst          ( rst_3                                                 ) ,

  .us_valid     ( mode != 3                                             ) ,
  .us_data      ( threshold_kind                                        ) ,
  .us_ready     (                                                       ) ,

  // We do not need to handle the valid here because
  // it should be identical to mode_valid_aux
  .ds_valid     (                                                       ) ,
  .ds_data      ( threshold_kind_aux                                    ) ,
  .ds_ready     ( core_s_axis_ready_aux || passthrough                  )
);

//////////////////////////////////////////////////////////////
// Register slices helping with compute pipelining
//////////////////////////////////////////////////////////////

always_comb
begin
    // According to the AXI-Stream protocol, the ready signal
    // might depend on valid, but the valid cannot depend on
    // the ready.
    //passthrough = mode_valid_aux && (mode_aux == 1 || mode_aux == 2);
    passthrough = mode_valid_aux && (mode_aux == 2);

    ml_valid = p5_valid;
    p5_ready = ml_ready;
    //p0_valid = mode_valid_aux && (mode_aux == 0 || mode_aux == 4);
    p0_valid = mode_valid_aux && (mode_aux == 0 || mode_aux == 4 || mode_aux == 1);
    p0_mode  = mode_valid_aux ? mode_aux : 3;
    p0_tkind = threshold_kind_aux;
    p0_tree_id_aux = tree_id_aux;
    p0_class_id_aux = class_id_aux;
    p0_input_leaf_value_aux = input_leaf_value_aux;
    p0_clause_aux = clause_aux;

    p4_ready = p5_ready || !p5_valid;
    p3_ready = p4_ready || !p4_valid;

    // The following bypasses stage 2
    //p2_ready = p3_ready || !p3_valid;
    p2_ready = p3_ready;
    p3_valid = p2_valid;

    // The following is disabled because
    // we are using `p1_ready_replicas`
    // instead
    //p1_ready = p2_ready || !p2_valid;

    // TODO: Maintain multiple p0_ready replicas as well
    p0_ready = p1_ready_replicas[0] || !p1_valid;

    core_s_axis_ready_aux = p0_ready;
end

reg   [31:0]                                        debug_num_completed_inputs;
reg   [31:0]                                        debug_num_completed_outputs;

always_ff @(posedge clk)
begin
    if (rst_3)
    begin
        debug_num_completed_inputs <= 0;
        debug_num_completed_outputs <= 0;
    end else begin
        if ((mode_aux == 0 || mode_aux == 4) && core_s_axis_ready_aux && mode_valid_aux)
        begin
            debug_num_completed_inputs <= debug_num_completed_inputs + 1;
        end

        if (ml_valid && ml_ready)
        begin
            debug_num_completed_outputs <= debug_num_completed_outputs + 1;
        end
    end
end

always_comb
begin
    for (int i = 0; i < NUM_CLAUSES; i++)
    begin
        for (int j = 0; j < NUM_VARS; j++)
        begin
            match_bits_aux[i][j] =                         (input_vars_aux_replicas[i / NUM_REPLICA_CONSUMERS][j] >= lower_thresholds[i][j]);
            match_bits_aux[i][j] = match_bits_aux[i][j] && (input_vars_aux_replicas[i / NUM_REPLICA_CONSUMERS][j] < higher_thresholds[i][j]);
        end
    end

    for (int i = 0; i < NUM_INPUT_REPLICAS; i++)
    begin
        for (int j = 0; j < NUM_REPLICA_CONSUMERS; j++)
        begin
            flattened_match_bits_aux[i][j * NUM_VARS +: NUM_VARS] = match_bits_aux[i * NUM_REPLICA_CONSUMERS + j];
            p2_match_bits_aux[i * NUM_REPLICA_CONSUMERS + j] = p2_flattened_match_bits_aux[i][j * NUM_VARS +: NUM_VARS];
        end
    end

    for (int i = 0; i < NUM_CLAUSES; i++)
    begin
        match_bits[i] = match_bits_dont_care_replicas[i / NUM_REPLICA_CONSUMERS] | p2_match_bits_aux[i];
    end
end

// This generate block achieves registered
// replication of the p1_ready signal.
// Notice that we do not need to connect
// all handshake and data signals.
generate
    pipe_adapter_mult #( 
      .WIDTH        ( NUM_REPLICA_CONSUMERS * NUM_VARS ),
      .NUM_SLICES   ( NUM_INPUT_STAGES       )
    )
    i_cc_pipe_mb ( 
      .clk          ( clk                                                   ) ,
      .rst          ( rst_3                                                 ) ,

      .us_valid     ( p1_valid                                              ) ,
      .us_data      ( flattened_match_bits_aux[0]                           ) ,
      .us_ready     ( p1_ready_replicas[0]                                  ) ,

      .ds_valid     ( p2_valid                                              ) ,
      .ds_data      ( p2_flattened_match_bits_aux[0]                        ) ,
      .ds_ready     ( p2_ready                                              )
    );

    for (i = 1; i < NUM_INPUT_REPLICAS; i++)
    begin
        pipe_adapter_mult #( 
          .WIDTH        ( NUM_REPLICA_CONSUMERS * NUM_VARS ),
          .NUM_SLICES   ( NUM_INPUT_STAGES       )
        )
        i_cc_pipe_mb ( 
          .clk          ( clk                                                   ) ,
          .rst          ( rst_3                                                 ) ,

          .us_valid     ( p1_valid                                              ) ,
          .us_data      ( flattened_match_bits_aux[i]                           ) ,
          .us_ready     ( p1_ready_replicas[i]                                  ) ,

          .ds_valid     (                                                       ) ,
          .ds_data      ( p2_flattened_match_bits_aux[i]                        ) ,
          .ds_ready     ( p2_ready                                              )
        );
    end
endgenerate

always_ff @(posedge clk)
begin
    // This breaks the long reset combinational path between
    // ap_done and this module
    rst_1 <= rst;
    rst_2 <= rst_1;
    rst_3 <= rst_2;

    p1_valid <= p1_valid;
    //p3_valid <= p3_valid;
    p4_valid <= p4_valid;
    p5_valid <= p5_valid;

    p1_input_leaf_value_aux <= p1_input_leaf_value_aux;
    p1_tree_id_aux          <= p1_tree_id_aux;
    p1_class_id_aux         <= p1_class_id_aux;

    if(rst_3) begin
        for (int i = 0; i < NUM_INPUT_REPLICAS; i++)
        begin
            match_bits_dont_care_replicas[i] <= 0;
        end

        enabled_clauses <= 0;

        p1_valid <= 0;
        //p3_valid <= 0;
        p4_valid <= 0;
        p5_valid <= 0;
    end else begin
        if (p0_ready)
        begin
            p1_valid                <= p0_valid && p0_mode != 1;
            p1_input_leaf_value_aux <= p0_input_leaf_value_aux;
            p1_tree_id_aux          <= p0_tree_id_aux;
            p1_class_id_aux         <= p0_class_id_aux;
        end

/*
        if (p2_ready)
        begin
            p3_valid <= p2_valid;
        end
*/

        if (p3_ready)
        begin
            p4_valid <= p3_valid;
        end

        if (p4_ready)
        begin
            p5_valid <= p4_valid;
        end

        for (int i = 0; i < NUM_INPUT_REPLICAS; i++)
        begin
            input_vars_aux_replicas[i] <= input_vars_aux_replicas[i];
            p1_clause_aux_replicas[i] <= p1_clause_aux_replicas[i];
            p1_mode_replicas[i] <= p1_mode_replicas[i];
            p1_tkind_replicas[i] <= p1_tkind_replicas[i];
            if (p0_ready)
            begin
                input_vars_aux_replicas[i] <= input_vars_aux;
                p1_clause_aux_replicas[i] <= p0_clause_aux;
                p1_mode_replicas[i] <= p0_mode;
                p1_tkind_replicas[i] <= p0_tkind;
            end
        end

        for (int i = 0; i < NUM_CLAUSES; i++)
        begin
            if (p1_clause_aux_replicas[i / NUM_REPLICA_CONSUMERS] == $clog2(NUM_CLAUSES)'(i))
            begin
                if (p1_mode_replicas[i / NUM_REPLICA_CONSUMERS] == 1)
                begin
                    if (p1_tkind_replicas[i / NUM_REPLICA_CONSUMERS] == 0)
                    begin
                        lower_thresholds[i] <= input_vars_aux_replicas[i / NUM_REPLICA_CONSUMERS];

                        tree_ids[i] <= p1_tree_id_aux;
                        class_ids[i] <= p1_class_id_aux;
                    end else begin
                        higher_thresholds[i] <= input_vars_aux_replicas[i / NUM_REPLICA_CONSUMERS];

                        leaf_values[i] <= p1_input_leaf_value_aux;

                        enabled_clauses[i] <= 1'b1;
                    end
                end
            end
        end

        for (int i = 0; i < NUM_CLAUSES; i++)
        begin
            match_lines_aux[i] <= match_lines_aux[i];
            if (p3_ready)
            begin
                match_lines_aux[i] <= &match_bits[i] && enabled_clauses[i];
            end
        end

/*
        // TODO-FIX: Implement and enable mode 0
        //           using the variable replicas
        if (mode_aux == 0 && mode_valid_aux)
        begin
            if (core_s_axis_ready_aux)
            begin
                match_lines <= match_lines_aux;
            end
        end
*/

        if (p4_ready)
        begin
            for (int i = 0; i < MAX_MATCHES; i++)
            begin
                match_leaves[i] <= 0;
                match_tree_ids[i] <= 0;
                match_class_ids[i] <= 0;
            end

            for (int i = 0; i < NUM_CLAUSES; i++)
            begin
                if (match_lines_aux[i])
                begin
                    // TODO: Ensure synthesis and implementation is not impaired
                    //       by an assumption that single entry from `match_leaves`
                    //       might be driven by multiple matches at execution.
                    //match_leaves[i >> $clog2(MIN_BRANCHES_PER_MATCH)] = leaf_values[i];
                    match_leaves[i / MIN_BRANCHES_PER_MATCH] <= leaf_values[i];
                    match_tree_ids[i / MIN_BRANCHES_PER_MATCH] <= tree_ids[i];
                    match_class_ids[i / MIN_BRANCHES_PER_MATCH] <= class_ids[i];
                end
            end
        end

        if (mode_aux == 2 && mode_valid_aux)
        begin
            for (int i = 0; i < NUM_INPUT_REPLICAS; i++)
            begin
                match_bits_dont_care_replicas[i][variable_aux] <= 1'b1;
            end
        end

    end
end
endmodule
