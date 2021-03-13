// RUN: npcomp-opt -rd-lower-next-func %s | FileCheck --dump-input=fail %s

// CHECK: rd.pipeline_def
"rd.pipeline_def"() ( {
  // TODO: HERE!!!!! ADD IN CHECKS!
  func @next(%arg0: !rd.Iterator) -> !rd.Dataset {
    %0 = rd.iterator_index %arg0 [0] : !rd.Iterator
    %1 = "rd.range.next"(%0) : (!rd.Iterator) -> !rd.Dataset
    %2 = rd.iterator_index %arg0 [1] : !rd.Iterator
    %3 = "rd.inline_map.next"(%1, %2) : (!rd.Dataset, !rd.Iterator) -> !rd.Dataset
    %4 = rd.iterator_index %arg0 [2] : !rd.Iterator
    %5 = "rd.filter.next"(%3, %4) : (!rd.Dataset, !rd.Iterator) -> !rd.Dataset
    return %5 : !rd.Dataset
  }
  rd.pipeline_def_terminator
}) { sym_name = "range_map_filter"} : () -> ()
