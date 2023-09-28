#!/bin/bash

DEVICE_LOC="${DEVICE_LOC:-83:00.0}"
PARTITION_XSABIN="${PARTITION_XSABIN:-/lib/firmware/xilinx/12c8fafb0632499db1c0c6676271b8a6/partition.xsabin}"

sudo env PATH=$PATH xbmgmt program --shell $PARTITION_XSABIN -d $DEVICE_LOC
