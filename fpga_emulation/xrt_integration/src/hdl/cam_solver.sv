`include "all_params.sv"

module cam_solver
#(
    parameter NUM_ROUTER_OUTPUTS = MAX_MATCHES,
    parameter NUM_ROUTER_INPUTS = MAX_MATCHES
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
    output logic signed [LEAF_VALUES_NUM_BITS-1:0]  match_leaves [NUM_ROUTER_OUTPUTS-1:0],
    output logic [TREE_ID_NUM_BITS-1:0]             match_tree_ids [NUM_ROUTER_OUTPUTS-1:0],
    output logic [CLASS_ID_NUM_BITS-1:0]            match_class_ids [NUM_ROUTER_OUTPUTS-1:0],
    input logic                                     ml_ready,
    output logic                                    ml_valid
);
    generate
        if (ATTACH_ROUTER_TO_CORE)
        begin
            logic [NUM_CLAUSES-1:0]                 aux_match_lines;
            logic signed [LEAF_VALUES_NUM_BITS-1:0] aux_match_leaves [NUM_ROUTER_INPUTS-1:0];
            logic [TREE_ID_NUM_BITS-1:0]            aux_match_tree_ids [NUM_ROUTER_INPUTS-1:0];
            logic [CLASS_ID_NUM_BITS-1:0]           aux_match_class_ids [NUM_ROUTER_INPUTS-1:0];
            logic                                   aux_ml_ready;
            logic                                   aux_ml_valid;

            cam_core #(
                .NUM_OUTPUTS(NUM_ROUTER_INPUTS)
            ) cam_core_inst (
                .clk                    (clk),
                .rst                    (rst),
                .clause                 (clause),
                .variable               (variable),
                .threshold_kind         (threshold_kind),
                .mode                   (mode),
                .input_vars             (input_vars),
                .core_s_axis_ready      (core_s_axis_ready),
                .match_lines            (match_lines),
                .match_leaves           (aux_match_leaves),
                .match_tree_ids         (aux_match_tree_ids),
                .match_class_ids        (aux_match_class_ids),
                .ml_valid               (aux_ml_valid),
                .ml_ready               (aux_ml_ready),
                .tree_id                (tree_id),
                .class_id               (class_id),
                .input_leaf_value       (input_leaf_value)
            );

            router #(
                .NUM_ROUTER_OUTPUTS(NUM_ROUTER_OUTPUTS)
            ) router_inst (
                .clk                    (clk),
                .rst                    (rst),
                .s_leaf_values          (aux_match_leaves),
                .s_class_ids            (aux_match_class_ids),
                .s_tree_ids             (aux_match_tree_ids),
                .s_ready                (aux_ml_ready),
                .m_valid                (ml_valid),
                .m_ready                (ml_ready),
                .m_leaf_values          (match_leaves),
                .m_class_ids            (match_class_ids),

                //TODO: Dynamically decide this
                .mode                   (aux_ml_valid ? 1 : 0)
            );
        end else begin
            cam_core
            cam_core_inst (
                .clk                    (clk),
                .rst                    (rst),
                .clause                 (clause),
                .variable               (variable),
                .threshold_kind         (threshold_kind),
                .mode                   (mode),
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
        end
    endgenerate
endmodule
