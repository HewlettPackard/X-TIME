`ifndef ALL_PARAMS_INCLUDE
`define ALL_PARAMS_INCLUDE

`include "params.sv"
`include "math.sv"

//parameter integer MAX_MATCHES = round_up_div(NUM_CLAUSES, MIN_BRANCHES_PER_MATCH);
parameter integer MAX_MATCHES = int'($ceil(NUM_CLAUSES / MIN_BRANCHES_PER_MATCH));

parameter NUM_CORES = 4;
parameter TOP_NUM_ROUTER_OUTPUTS = 1;
`endif
