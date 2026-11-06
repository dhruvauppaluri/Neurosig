# Invoked by run_c5g.py. All paths are absolute Windows paths.
proc rd {mp addr} {
    set value [lindex [master_read_32 $mp $addr 1] 0]
    expr {$value & 0xffffffff}
}
proc wr {mp addr value} { master_write_32 $mp $addr [list $value] }
proc check_error {mp} {
    set err [rd $mp 0x0c]
    if {$err != 0} { error [format "FPGA error flags 0x%08x" $err] }
}
proc wait_idle {mp} {
    for {set i 0} {$i < 100000} {incr i} {
        check_error $mp
        if {[rd $mp 0x08] & 4} { return }
        after 1
    }
    error "FPGA did not return to idle"
}
proc drain {mp out} {
    while {1} {
        set occupancy [rd $mp 0x28]
        if {$occupancy > $::nsg_max_fifo} { set ::nsg_max_fifo $occupancy }
        if {$occupancy == 0} { break }
        set idx [rd $mp 0x2c]
        set data [rd $mp 0x30]
        set seq [rd $mp 0x34]
        if {!(($data >> 29) & 1)} { error "invalid event record" }
        set ch [expr {($data >> 24) & 31}]
        set code [expr {$data & 0xffffff}]
        set flags [expr {($data >> 30) & 3}]
        if {$code & 0x800000} { set code [expr {$code - 0x1000000}] }
        puts $out "$seq,$idx,$ch,$code,$flags"
        flush $out
        wr $mp 0x3c 1
    }
}

set paths [get_service_paths master]
if {[llength $paths] != 1} { error "expected one JTAG master, found: $paths" }
set mp [claim_service master [lindex $paths 0] neurosig]
if {[rd $mp 0x00] != 0x4e534734} { error "wrong JTAG device or bitstream" }
set out [open $::env(NSG_EVENTS_CSV) a]
set f [open $::env(NSG_WORDS_BIN) rb]
fconfigure $f -translation binary -encoding binary
set thresholds [open $::env(NSG_THRESHOLDS) r]
if {$::env(NSG_RESUME) == 0} {
    wait_idle $mp
    wr $mp 0x40 0x52455354
    wait_idle $mp
    wr $mp 0x10 $::env(NSG_SESSION)
    for {set c 0} {$c < 32} {incr c} {
        if {[gets $thresholds line] < 0} { error "only $c thresholds supplied" }
        scan $line %x value
        wr $mp [expr {0x100 + 4*$c}] $value
    }
} else {
    wait_idle $mp
    if {[rd $mp 0x10] != $::env(NSG_SESSION)} { error "session mismatch on resume" }
}
close $thresholds
set last [rd $mp 0x20]
set ::nsg_max_fifo 0
set block 0
set sample_count 0
drain $mp $out
while {1} {
    set raw [read $f 4096]
    if {[string length $raw] == 0} { break }
    if {[string length $raw] % 4} { error "short packed sample word" }
    set n [expr {[string length $raw] / 4}]
    binary scan $raw i* words
    if {$last == 0xffffffff || $block > $last} {
        wait_idle $mp
        master_write_32 $mp 0x1000 $words
        wr $mp 0x14 $block
        wr $mp 0x18 $n
        wr $mp 0x1c [zlib crc32 $raw]
        wr $mp 0x40 0x434f4d54
        wait_idle $mp
        if {[rd $mp 0x20] != $block} { error "block $block not acknowledged" }
        drain $mp $out
    }
    incr sample_count $n
    incr block
}
close $f
if {[rd $mp 0x24] != $sample_count} { error "FPGA sample count mismatch" }
wr $mp 0x40 0x464c5348
wait_idle $mp
drain $mp $out
check_error $mp
puts "NSG_SUCCESS samples=$sample_count max_fifo=$::nsg_max_fifo errors=[rd $mp 0x0c] feed_cycles=[rd $mp 0x44] check_cycles=[rd $mp 0x48] fifo_peak_hw=[rd $mp 0x4c] emitted=[rd $mp 0x50]"
close $out
close_service master $mp
