set_clock_groups -name kill_false_path_1 -asynchronous -group [get_clocks io_clk_pcie_user_00] -group [get_clocks -of_objects [get_pins level0_i/blp/blp_i/ss_hif/inst/pcie/inst/bd_39ab_pcie_0_gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]]
set_clock_groups -name kill_false_path_2 -asynchronous -group [get_clocks level0_i/blp/blp_i/ss_cmp/inst/jtag_fallback/inst/bs_switch/inst/BSCAN_SWITCH.N_EXT_BSCAN.bscan_inst/SERIES7_BSCAN.bscan_inst/INTERNAL_TCK] -group [get_clocks -of_objects [get_pins level0_i/blp/blp_i/ss_hif/inst/clkwiz_level0_periph/inst/plle4_adv_inst/CLKOUT1]]