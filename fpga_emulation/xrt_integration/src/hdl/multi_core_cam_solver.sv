`include "all_params.sv"

module multi_core_cam_solver
#(
    parameter integer TREE_HEIGHT = 1,
    parameter integer NUM_ROUTER_OUTPUTS = 1,

    // The following parameters should not be directly
    // based on the total number of cores in the system.
    // Rather, they should reflect the number of cores
    // under this particular NoC node.
    parameter integer NUM_MODE_CHANNELS = 2 ** TREE_HEIGHT,
    parameter integer HIGHEST_LOW_MODE_CHANNEL = NUM_MODE_CHANNELS / 2
)
(
    input wire                                          clk,
    input wire                                          rst,
    input logic [$clog2(NUM_CLAUSES)-1:0]               clause,
    input logic [$clog2(NUM_VARS)-1:0]                  variable,
    input wire                                          threshold_kind,

    // This is the only input that needs to be replicated as
    // a function of `TREE_HEIGHT` because all other inputs
    // are either driven by broadcast or might be selectively
    // applied to only a single child module at a time as a
    // function of `mode`.
    input logic [2:0]                                   mode [NUM_MODE_CHANNELS-1:0],

    input wire [VAR_WIDTH-1:0]                          input_vars [NUM_VARS-1:0],
    input logic signed [LEAF_VALUES_NUM_BITS-1:0]       input_leaf_value,
    input logic [TREE_ID_NUM_BITS-1:0]                  tree_id,
    input logic [CLASS_ID_NUM_BITS-1:0]                 class_id,
    output logic                                        core_s_axis_ready,

    output logic [NUM_CLAUSES-1:0]                      match_lines,

    // TODO: Dynamically determine the size of these arrays
    //       to support multi-class classification.
    output logic signed [LEAF_VALUES_NUM_BITS-1:0]      match_leaves [NUM_ROUTER_OUTPUTS-1:0],
    output logic [TREE_ID_NUM_BITS-1:0]                 match_tree_ids [NUM_ROUTER_OUTPUTS-1:0],
    output logic [CLASS_ID_NUM_BITS-1:0]                match_class_ids [NUM_ROUTER_OUTPUTS-1:0],

    input logic                                         ml_ready,
    output logic                                        ml_valid
);
    generate
        if (TREE_HEIGHT == 0)
        begin
            cam_solver #(
                .NUM_ROUTER_OUTPUTS(NUM_ROUTER_OUTPUTS)
            ) cam_solver_inst (
                .clk                    (clk),
                .rst                    (rst),
                .clause                 (clause),
                .variable               (variable),
                .threshold_kind         (threshold_kind),
                .mode                   (mode[0]),
                .input_vars             (input_vars),
                .match_lines            (match_lines),
                .match_tree_ids         (match_tree_ids),
                .match_class_ids        (match_class_ids),
                .tree_id                (tree_id),
                .class_id               (class_id),
                .ml_valid               (ml_valid),
                .ml_ready               (ml_ready),
                .core_s_axis_ready      (core_s_axis_ready),
                .input_leaf_value       (input_leaf_value),
                .match_leaves           (match_leaves)
            );
        end else begin
            localparam NUM_ROUTER_INPUTS = 2;

            logic signed [LEAF_VALUES_NUM_BITS-1:0]     aux_router_match_leaves[NUM_ROUTER_INPUTS-1:0];
            logic [CLASS_ID_NUM_BITS-1:0]               aux_router_match_class_ids[NUM_ROUTER_INPUTS-1:0];

            logic signed [LEAF_VALUES_NUM_BITS-1:0]     aux_solver_match_leaves[NUM_ROUTER_INPUTS-1:0][NUM_ROUTER_OUTPUTS-1:0];
            logic [CLASS_ID_NUM_BITS-1:0]               aux_solver_match_class_ids[NUM_ROUTER_INPUTS-1:0][NUM_ROUTER_OUTPUTS-1:0];

            logic                                       aux_ml_ready;
            logic                                       aux_ml_valid;

            initial
            begin
                assert (NUM_ROUTER_INPUTS >= 2 * NUM_ROUTER_OUTPUTS) else $fatal("NUM_ROUTER_INPUTS should not be lower than (2 * NUM_ROUTER_OUTPUTS)");
            end

            genvar i;
            genvar j;

            if (BINARY_CLASSIFICATION_ONLY)
            begin
                for (i = 0; i < NUM_ROUTER_OUTPUTS; i++)
                begin
                    assign match_tree_ids[i] = 0;
                end
            end

            for (i = 0; i < 2; i++)
            begin
                for (j = 0; j < NUM_ROUTER_OUTPUTS; j++)
                begin
                    assign aux_router_match_leaves[i * NUM_ROUTER_OUTPUTS + j] = aux_solver_match_leaves[i][j];
                    assign aux_router_match_class_ids[i * NUM_ROUTER_OUTPUTS + j] = aux_solver_match_class_ids[i][j];
                end
            end

            for (i = NUM_ROUTER_OUTPUTS * 2; i < NUM_ROUTER_INPUTS; i++)
            begin
                assign aux_router_match_leaves[i] = 0;
                assign aux_router_match_class_ids[i] = 0;
            end

            // TODO: Set the following parameters dynamically once
            //       we start supporting multi-class classification
            router #(
                .NUM_ROUTER_INPUTS(NUM_ROUTER_INPUTS),
                .NUM_ROUTER_OUTPUTS(NUM_ROUTER_OUTPUTS)
            ) router_inst (
                .clk                    (clk),
                .rst                    (rst),
                .s_leaf_values          (aux_router_match_leaves),
                .s_class_ids            (aux_router_match_class_ids),
                .s_tree_ids             (),
                .s_ready                (aux_ml_ready),
                .m_valid                (ml_valid),
                .m_ready                (ml_ready),
                .m_leaf_values          (match_leaves),
                .m_class_ids            (match_class_ids),

                //TODO: Dynamically decide this
                .mode                   (aux_ml_valid ? 1 : 0)
            );

            multi_core_cam_solver #(
                .TREE_HEIGHT(TREE_HEIGHT - 1)
            ) multic_core_cam_solver_inst_0 (
                .clk                    (clk),
                .rst                    (rst),
                .clause                 (clause),
                .variable               (variable),
                .threshold_kind         (threshold_kind),
                .mode                   (mode[NUM_MODE_CHANNELS-1:HIGHEST_LOW_MODE_CHANNEL]),
                .input_vars             (input_vars),
                .core_s_axis_ready      (core_s_axis_ready),
                .match_lines            (),
                .match_leaves           (aux_solver_match_leaves[0]),
                .match_tree_ids         (),
                .match_class_ids        (aux_solver_match_class_ids[0]),
                .ml_valid               (aux_ml_valid),
                .ml_ready               (aux_ml_ready),
                .tree_id                (tree_id),
                .class_id               (class_id),
                .input_leaf_value       (input_leaf_value)
            );

            multi_core_cam_solver #(
                .TREE_HEIGHT(TREE_HEIGHT - 1)
            ) multic_core_cam_solver_inst_1 (
                .clk                    (clk),
                .rst                    (rst),
                .clause                 (clause),
                .variable               (variable),
                .threshold_kind         (threshold_kind),
                .mode                   (mode[HIGHEST_LOW_MODE_CHANNEL-1:0]),
                .input_vars             (input_vars),
                .match_lines            (),
                .match_leaves           (aux_solver_match_leaves[1]),
                .match_tree_ids         (),
                .match_class_ids        (aux_solver_match_class_ids[1]),
                .tree_id                (tree_id),
                .class_id               (class_id),
                .input_leaf_value       (input_leaf_value),

                // The following signals are left unconnected
                // because their state can always be inferred
                // by the analogous signals from the primary
                // solver instance.
                .core_s_axis_ready      (),                     
                .ml_valid               (),

                // The following input must remain connected
                // because the solver uses it to determine
                // when to update output values.
                .ml_ready               (aux_ml_ready)
            );
        end
    endgenerate
endmodule
