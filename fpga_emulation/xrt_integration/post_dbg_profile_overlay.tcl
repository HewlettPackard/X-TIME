set RUNDIR [ file dirname [ file normalize [ info script ] ] ]
#read_xdc $RUNDIR/connect_debug_core.xdc
#set_property used_in_implementation TRUE [get_files $RUNDIR/connect_debug_core.xdc]
#set_property PROCESSING_ORDER EARLY [get_files $RUNDIR/connect_debug_core.xdc]

#read_xdc $RUNDIR/remove_unsafe_false_paths.xdc
#set_property used_in_implementation TRUE [get_files $RUNDIR/remove_unsafe_false_paths.xdc]
#set_property PROCESSING_ORDER LATE [get_files $RUNDIR/remove_unsafe_false_paths.xdc]
