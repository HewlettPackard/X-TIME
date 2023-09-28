#!/bin/bash

rm -rf rtl_vadd_hw_debug
OPT_LEVEL=-O3 make rtl_vadd_hw_debug
#./rtl_vadd_hw_debug build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/vadd.xclbin /home/moraisl/Summer-CAMp/artifacts/quant_datasets/ex01
#./rtl_vadd_hw_debug build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/vadd.xclbin /home/moraisl/Summer-CAMp/artifacts/quant_datasets/ex02-binary-6t-256b-3v
#./rtl_vadd_hw_debug build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/vadd.xclbin /home/moraisl/Summer-CAMp/artifacts/quant_datasets/ex03-binary-100t-256b-120v
#./rtl_vadd_hw_debug build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/vadd.xclbin /home/moraisl/Summer-CAMp/artifacts/quant_datasets/ex04-binary-256t-4b-29v
#./rtl_vadd_hw_debug build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/vadd.xclbin /home/moraisl/Summer-CAMp/artifacts/quant_datasets/ex05-churn-modelling-512t-12f-10Kq
./rtl_vadd_hw_debug build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/vadd.xclbin /home/moraisl/Summer-CAMp/artifacts/quant_datasets/ex06-telco-churn-256t-19f-7032q
#./rtl_vadd_hw_debug build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/vadd.xclbin /home/moraisl/Summer-CAMp/artifacts/quant_datasets/ex07-churn-modelling-8bit-512t-12f-10Kq
