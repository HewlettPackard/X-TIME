`ifndef PARAMS_INCLUDE
`define PARAMS_INCLUDE

parameter C_AXIS_TDATA_WIDTH = 512;                         

// It is required that NUM_VARS * VAR_WIDTH <= (C_AXIS_TDATA_WIDTH - max(LEAF_VALUES_NUM_BITS, CLASS_ID_NUM_BITS + TREE_ID_NUM_BITS))
parameter NUM_VARS = 30;                                                     // typical: 128
parameter integer NUM_CLAUSES = 16;                                          // 512
parameter integer MIN_BRANCHES_PER_MATCH = 8;

parameter VAR_WIDTH = 4;                                                    
//parameter VAR_WIDTH = 4;                                                    
parameter NUM_STATES = 1 << VAR_WIDTH;                                      // 16
parameter MAX_STATE = NUM_STATES - 1;                                       // 15

// This controls don't care alignment,
// determines maximum leaf value precision,
// and influences the maximum number of
// active don't cares in the system.
parameter MACRO_VAR_WIDTH = 32;                                             

parameter QUERY_ID_WIDTH = 32;
parameter LEAF_VALUES_NUM_BITS = 32;
parameter TREE_ID_NUM_BITS = 22;
parameter CLASS_ID_NUM_BITS = 10;
parameter C_MAX_LENGTH_WIDTH = 32;

// It is required that `NUM_CLAUSES % NUM_REPLICA_CONSUMERS == 0`
parameter NUM_REPLICA_CONSUMERS = 16;
parameter NUM_CORE_INPUT_STAGES = 3;

parameter NOC_AXI_OUTPUT = 1;
parameter BINARY_CLASSIFICATION_ONLY = 1;
parameter ATTACH_ROUTER_TO_CORE = 1;
`endif
