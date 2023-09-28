#!/bin/bash

vivado -mode batch -source ./scripts/gen_xo_do_not_package.tcl -notrace -tclargs ila_proj.xo vadd hw none xilinx_u250_gen3x16_xdma_4_1_202210_1
