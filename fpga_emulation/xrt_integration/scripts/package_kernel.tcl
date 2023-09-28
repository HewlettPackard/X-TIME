#
# Copyright (C) 2019-2021 Xilinx, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may
# not use this file except in compliance with the License. A copy of the
# License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#

set path_to_hdl "./src/hdl"
set path_to_packaged "./packaged_kernel_${suffix}"
set path_to_tmp_project "./tmp_kernel_pack_${suffix}"

create_project -force kernel_pack $path_to_tmp_project 
add_files -norecurse [glob $path_to_hdl/*.v $path_to_hdl/*.sv]

set_property board_part xilinx.com:au250:part0:1.3 [current_project]

create_ip -name ila -vendor xilinx.com -library ip -version 6.2 -module_name ila_0
set_property -dict [list CONFIG.C_PROBE6_WIDTH {512} CONFIG.C_PROBE3_WIDTH {512} CONFIG.C_NUM_OF_PROBES {7} CONFIG.C_EN_STRG_QUAL {1} CONFIG.C_INPUT_PIPE_STAGES {2} CONFIG.C_ADV_TRIGGER {true} CONFIG.ALL_PROBE_SAME_MU_CNT {4} CONFIG.C_PROBE6_MU_CNT {4} CONFIG.C_PROBE5_MU_CNT {4} CONFIG.C_PROBE4_MU_CNT {4} CONFIG.C_PROBE3_MU_CNT {4} CONFIG.C_PROBE2_MU_CNT {4} CONFIG.C_PROBE1_MU_CNT {4} CONFIG.C_PROBE0_MU_CNT {4}] [get_ips ila_0]
set_property -dict [list CONFIG.C_TRIGIN_EN {true}] [get_ips ila_0]
generate_target {instantiation_template} [get_files ila_0.xci]
set_property generate_synth_checkpoint false [get_files ila_0.xci]

create_ip -name ila -vendor xilinx.com -library ip -version 6.2 -module_name ila_1
set_property -dict [list CONFIG.C_NUM_OF_PROBES {20} CONFIG.C_EN_STRG_QUAL {1} CONFIG.C_INPUT_PIPE_STAGES {2} CONFIG.C_ADV_TRIGGER {true} CONFIG.C_PROBE19_MU_CNT {3} CONFIG.C_PROBE18_MU_CNT {3} CONFIG.C_PROBE17_MU_CNT {3} CONFIG.C_PROBE16_MU_CNT {3} CONFIG.C_PROBE15_MU_CNT {3} CONFIG.C_PROBE14_MU_CNT {3} CONFIG.C_PROBE13_MU_CNT {3} CONFIG.C_PROBE12_MU_CNT {3} CONFIG.C_PROBE11_MU_CNT {3} CONFIG.C_PROBE10_MU_CNT {3} CONFIG.C_PROBE9_MU_CNT {3} CONFIG.C_PROBE8_MU_CNT {3} CONFIG.C_PROBE7_MU_CNT {3} CONFIG.C_PROBE6_MU_CNT {3} CONFIG.C_PROBE5_MU_CNT {3} CONFIG.C_PROBE4_MU_CNT {3} CONFIG.C_PROBE3_MU_CNT {3} CONFIG.C_PROBE2_MU_CNT {3} CONFIG.C_PROBE1_MU_CNT {3} CONFIG.C_PROBE0_MU_CNT {3} CONFIG.ALL_PROBE_SAME_MU_CNT {3}] [get_ips ila_1]
set_property -dict [list CONFIG.C_PROBE18_WIDTH {16} CONFIG.C_PROBE17_WIDTH {16} CONFIG.C_PROBE15_WIDTH {16} CONFIG.C_PROBE14_WIDTH {16} CONFIG.C_PROBE13_WIDTH {32} CONFIG.C_PROBE12_WIDTH {32} CONFIG.C_PROBE9_WIDTH {16} CONFIG.C_PROBE8_WIDTH {16} CONFIG.C_PROBE7_WIDTH {16} CONFIG.C_PROBE6_WIDTH {16} CONFIG.C_PROBE5_WIDTH {16} CONFIG.C_PROBE4_WIDTH {16} CONFIG.C_PROBE3_WIDTH {16} CONFIG.C_PROBE2_WIDTH {16} CONFIG.C_PROBE1_WIDTH {16} CONFIG.C_PROBE0_WIDTH {16}] [get_ips ila_1]
set_property -dict [list CONFIG.C_TRIGOUT_EN {true} CONFIG.C_TRIGIN_EN {false}] [get_ips ila_1]
set_property -dict [list CONFIG.C_PROBE22_WIDTH {512} CONFIG.C_NUM_OF_PROBES {23} CONFIG.C_PROBE22_MU_CNT {3} CONFIG.C_PROBE21_MU_CNT {3} CONFIG.C_PROBE20_MU_CNT {3}] [get_ips ila_1]
set_property -dict [list CONFIG.C_NUM_OF_PROBES {24} CONFIG.C_PROBE23_MU_CNT {3}] [get_ips ila_1]
generate_target {instantiation_template} [get_files ila_1.xci]
set_property generate_synth_checkpoint false [get_files ila_0.xci]

create_ip -name axis_register_slice -vendor xilinx.com -library ip -version 1.1 -module_name axis_register_slice_0
set_property -dict [list CONFIG.TDATA_NUM_BYTES {64} CONFIG.REG_CONFIG {8} CONFIG.HAS_TREADY {1} CONFIG.HAS_TLAST {0}] [get_ips axis_register_slice_0]
generate_target {instantiation_template} [get_files /tmp/project_1/project_1.srcs/sources_1/ip/axis_register_slice_0/axis_register_slice_0.xci]

create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name axi_register_slice_0
set_property -dict [list CONFIG.READ_WRITE_MODE {WRITE_ONLY} CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {512} CONFIG.REG_AW {1} CONFIG.REG_W {1} CONFIG.REG_B {1} CONFIG.SUPPORTS_NARROW_BURST {0} CONFIG.HAS_BURST {0} CONFIG.HAS_LOCK {0} CONFIG.HAS_CACHE {0} CONFIG.HAS_REGION {0} CONFIG.HAS_QOS {0} CONFIG.HAS_PROT {0} CONFIG.HAS_WSTRB {1} CONFIG.HAS_BRESP {0} CONFIG.HAS_RRESP {0} CONFIG.MAX_BURST_LENGTH {256} CONFIG.NUM_READ_OUTSTANDING {0} CONFIG.NUM_WRITE_OUTSTANDING {32}] [get_ips axi_register_slice_0]
generate_target {instantiation_template} [get_files /tmp/project_1/project_1.srcs/sources_1/ip/axi_register_slice_0/axi_register_slice_0.xci]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

set_property top krnl_vadd_rtl [current_fileset]

ipx::package_project -root_dir $path_to_packaged -vendor xilinx.com -library RTLKernel -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core $path_to_packaged/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory $path_to_packaged $path_to_packaged/component.xml

set core [ipx::current_core]

set_property core_revision 2 $core
foreach up [ipx::get_user_parameters] {
  ipx::remove_user_parameter [get_property NAME $up] $core
}
ipx::associate_bus_interfaces -busif m_axi_gmem -clock ap_clk $core
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk $core

set mem_map    [::ipx::add_memory_map -quiet "s_axi_control" $core]
set addr_block [::ipx::add_address_block -quiet "reg0" $mem_map]

set reg      [::ipx::add_register "CTRL" $addr_block]
  set_property description    "Control signals"    $reg
  set_property address_offset 0x000 $reg
  set_property size           32    $reg
set field [ipx::add_field AP_START $reg]
  set_property ACCESS {read-write} $field
  set_property BIT_OFFSET {0} $field
  set_property BIT_WIDTH {1} $field
  set_property DESCRIPTION {Control signal Register for 'ap_start'.} $field
  set_property MODIFIED_WRITE_VALUE {modify} $field
set field [ipx::add_field AP_DONE $reg]
  set_property ACCESS {read-only} $field
  set_property BIT_OFFSET {1} $field
  set_property BIT_WIDTH {1} $field
  set_property DESCRIPTION {Control signal Register for 'ap_done'.} $field
  set_property READ_ACTION {modify} $field
set field [ipx::add_field AP_IDLE $reg]
  set_property ACCESS {read-only} $field
  set_property BIT_OFFSET {2} $field
  set_property BIT_WIDTH {1} $field
  set_property DESCRIPTION {Control signal Register for 'ap_idle'.} $field
  set_property READ_ACTION {modify} $field
set field [ipx::add_field AP_READY $reg]
  set_property ACCESS {read-only} $field
  set_property BIT_OFFSET {3} $field
  set_property BIT_WIDTH {1} $field
  set_property DESCRIPTION {Control signal Register for 'ap_ready'.} $field
  set_property READ_ACTION {modify} $field
set field [ipx::add_field RESERVED_1 $reg]
  set_property ACCESS {read-only} $field
  set_property BIT_OFFSET {4} $field
  set_property BIT_WIDTH {3} $field
  set_property DESCRIPTION {Reserved.  0s on read.} $field
  set_property READ_ACTION {modify} $field
set field [ipx::add_field AUTO_RESTART $reg]
  set_property ACCESS {read-write} $field
  set_property BIT_OFFSET {7} $field
  set_property BIT_WIDTH {1} $field
  set_property DESCRIPTION {Control signal Register for 'auto_restart'.} $field
  set_property MODIFIED_WRITE_VALUE {modify} $field
set field [ipx::add_field RESERVED_2 $reg]
  set_property ACCESS {read-only} $field
  set_property BIT_OFFSET {8} $field
  set_property BIT_WIDTH {24} $field
  set_property DESCRIPTION {Reserved.  0s on read.} $field
  set_property READ_ACTION {modify} $field

set reg      [::ipx::add_register "GIER" $addr_block]
  set_property description    "Global Interrupt Enable Register"    $reg
  set_property address_offset 0x004 $reg
  set_property size           32    $reg

set reg      [::ipx::add_register "IP_IER" $addr_block]
  set_property description    "IP Interrupt Enable Register"    $reg
  set_property address_offset 0x008 $reg
  set_property size           32    $reg

set reg      [::ipx::add_register "IP_ISR" $addr_block]
  set_property description    "IP Interrupt Status Register"    $reg
  set_property address_offset 0x00C $reg
  set_property size           32    $reg

set reg      [::ipx::add_register -quiet "a" $addr_block]
  set_property address_offset 0x010 $reg
  set_property size           [expr {8*8}]   $reg
  set regparam [::ipx::add_register_parameter -quiet {ASSOCIATED_BUSIF} $reg] 
  set_property value m_axi_gmem $regparam 

set reg      [::ipx::add_register -quiet "b" $addr_block]
  set_property address_offset 0x01C $reg
  set_property size           [expr {8*8}]   $reg
  set regparam [::ipx::add_register_parameter -quiet {ASSOCIATED_BUSIF} $reg] 
  set_property value m_axi_gmem $regparam 

set reg      [::ipx::add_register -quiet "c" $addr_block]
  set_property address_offset 0x028 $reg
  set_property size           [expr {8*8}]   $reg
  set regparam [::ipx::add_register_parameter -quiet {ASSOCIATED_BUSIF} $reg] 
  set_property value m_axi_gmem $regparam 

set reg      [::ipx::add_register -quiet "length_r" $addr_block]
  set_property address_offset 0x034 $reg
  set_property size           [expr {4*8}]   $reg


set_property slave_memory_map_ref "s_axi_control" [::ipx::get_bus_interfaces -of $core "s_axi_control"]

set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} $core
set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core
set_property supported_families { } $core
set_property auto_family_support_level level_2 $core
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity -kernel $core
ipx::save_core $core
close_project -delete
