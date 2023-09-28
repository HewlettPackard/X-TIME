#!/bin/bash

build_path=.

if [ -n "$1" ];
then
	build_path=$1
fi

rm -f build.tar.gz
tar -czf build.tar.gz "$build_path"/build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/
md5sum build.tar.gz
cp build.tar.gz /tmp/
