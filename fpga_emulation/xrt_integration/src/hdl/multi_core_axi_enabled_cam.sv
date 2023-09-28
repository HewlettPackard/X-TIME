`include "all_params.sv"

module multi_core_axi_enabled_cam
#(
    parameter NUM_ROUTER_OUTPUTS = 1,
    parameter NUM_MODE_CHANNELS = NUM_CORES
)
(
    input wire                                  clk,
    input wire                                  rst,

    input logic                                 s_axis_tvalid,
    output logic                                s_axis_tready,
    input logic  [C_AXIS_TDATA_WIDTH-1:0]       s_axis_tdata,
    //input logic  [C_AXIS_TDATA_WIDTH/8-1:0]     s_axis_tkeep,
    //input logic                                 s_axis_tlast,

    output wire                                 m_axis_tvalid,
    input  wire                                 m_axis_tready,
    output wire  [C_AXIS_TDATA_WIDTH-1:0]       m_axis_tdata,

    output logic [C_MAX_LENGTH_WIDTH-1:0]       num_out_tx,
    output logic                                out_pulse,
    output logic                                finished,
    
    output logic                                trig_out,
    input logic                                 trig_out_ack
);

    logic [$clog2(NUM_CLAUSES)-1:0]             clause;
    logic [$clog2(NUM_VARS)-1:0]                variable;
    logic [2:0]                                 mode [NUM_MODE_CHANNELS-1:0];
    logic [VAR_WIDTH-1:0]                       input_vars [NUM_VARS-1:0];
    logic [NUM_CLAUSES-1:0]                     match_lines;
    logic signed [LEAF_VALUES_NUM_BITS-1:0]     leaf_value;
    logic signed [LEAF_VALUES_NUM_BITS-1:0]     match_leaves [NUM_ROUTER_OUTPUTS-1:0];
    logic [TREE_ID_NUM_BITS-1:0]                match_tree_ids [NUM_ROUTER_OUTPUTS-1:0];
    logic [CLASS_ID_NUM_BITS-1:0]               match_class_ids [NUM_ROUTER_OUTPUTS-1:0];
    logic [TREE_ID_NUM_BITS-1:0]                tree_id;
    logic [CLASS_ID_NUM_BITS-1:0]               class_id;
    logic                                       threshold_kind;
    logic                                       ml_valid;
    logic                                       ml_ready;
    logic                                       core_s_axis_ready;

multi_core_axi_io_helper #(
    .NUM_MODE_CHANNELS(NUM_MODE_CHANNELS),
    .NUM_ROUTER_OUTPUTS(NUM_ROUTER_OUTPUTS)
) multi_core_axi_io_helper_inst (
    .clk                    (clk),
    .rst                    (rst),

    .s_axis_tvalid          (s_axis_tvalid),
    .s_axis_tready          (s_axis_tready),
    .s_axis_tdata           (s_axis_tdata),
    //.s_axis_tkeep             (s_axis_tkeep),
    //.s_axis_tlast             (s_axis_tlast),

    .m_axis_tvalid          (m_axis_tvalid),
    .m_axis_tready          (m_axis_tready),
    .m_axis_tdata           (m_axis_tdata),

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
    .out_pulse              (out_pulse),
    .num_out_tx             (num_out_tx),
    .finished               (finished),
    
    .trig_out               (trig_out),
    .trig_out_ack           (trig_out_ack),

    .leaf_value             (leaf_value),
    .match_leaves           (match_leaves)
);

multi_core_cam_solver #(
    .TREE_HEIGHT($floor($clog2(NUM_CORES))),
    .NUM_ROUTER_OUTPUTS(NUM_ROUTER_OUTPUTS)
) multi_core_cam_solver_inst (
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
    .input_leaf_value       (leaf_value),
    .match_leaves           (match_leaves)
);


endmodule
