`include "all_params.sv"

module router
#(
    parameter NUM_ROUTER_INPUTS = MAX_MATCHES,
    parameter NUM_ROUTER_OUTPUTS = MAX_MATCHES
)
(
    input wire                                      clk,
    input wire                                      rst,

    input logic                                     mode,
    input logic signed [LEAF_VALUES_NUM_BITS-1:0]   s_leaf_values [NUM_ROUTER_INPUTS-1:0],
    input logic [CLASS_ID_NUM_BITS-1:0]             s_class_ids   [NUM_ROUTER_INPUTS-1:0],
    input logic [TREE_ID_NUM_BITS-1:0]              s_tree_ids    [NUM_ROUTER_INPUTS-1:0],
    output logic                                    s_ready,
    output logic                                    m_valid,
    input logic                                     m_ready,

    output logic signed [LEAF_VALUES_NUM_BITS-1:0]  m_leaf_values      [NUM_ROUTER_OUTPUTS-1:0],
    output logic [CLASS_ID_NUM_BITS-1:0]            m_class_ids        [NUM_ROUTER_OUTPUTS-1:0]
);
    generate
        logic signed [LEAF_VALUES_NUM_BITS-1:0] partial_accumulation [NUM_ROUTER_INPUTS-1:0][NUM_ROUTER_OUTPUTS-1:0];

        // TODO: Implement support for multi-class classification
        if (BINARY_CLASSIFICATION_ONLY)
        begin
            always_comb
            begin
                partial_accumulation[0][0] = s_leaf_values[0];

                for (int i = 1; i < NUM_ROUTER_INPUTS; i++)
                begin
                    partial_accumulation[i][0] = s_leaf_values[i] + partial_accumulation[i-1][0];
                end
            end
        end

        always_comb
        begin
            s_ready = m_ready || !m_valid;
        end

        reg   [31:0]                                        debug_num_completed_inputs;
        reg   [31:0]                                        debug_num_completed_outputs;

        always_ff @(posedge clk)
        begin
            if (rst)
            begin
                debug_num_completed_inputs <= 0;
                debug_num_completed_outputs <= 0;
            end else begin
                if ((mode == 1) && s_ready)
                begin
                    debug_num_completed_inputs <= debug_num_completed_inputs + 1;
                end

                if (m_valid && m_ready)
                begin
                    debug_num_completed_outputs <= debug_num_completed_outputs + 1;
                end
            end
        end

        always_ff @(posedge clk)
        begin
            if (rst)
            begin
                m_valid <= 0;
            end else begin
                if (s_ready)
                begin
                    for (int i = 0; i < NUM_ROUTER_OUTPUTS; i++)
                    begin
                        m_leaf_values[i] <= 0;
                        m_class_ids[i]   <= 0;
                    end

                    m_leaf_values[0] <= partial_accumulation[NUM_ROUTER_INPUTS-1][0];
                    m_class_ids[0]   <= partial_accumulation[NUM_ROUTER_INPUTS-1][0] > 0 ? 1 : 0;

                    m_valid          <= (mode == 1);
                end else begin
                    m_valid          <= m_valid;
                end
            end
        end
    endgenerate
endmodule
