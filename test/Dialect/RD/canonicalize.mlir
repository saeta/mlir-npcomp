// RUN: npcomp-opt -split-input-file %s | npcomp-opt -canonicalize | FileCheck --dump-input=fail %s

// CHECK-LABEL: func @simple_range
func @simple_range(%start: i64, %end: i64) -> !rd.Dataset {
  // CHECK: %[[DS:.*]] = rd.range
  %0 = rd.range %start to %end  : (i64, i64) -> !rd.Dataset
  // CHECK: return %[[DS]]
  return %0 : !rd.Dataset
}

// CHECK-LABEL: func @create_and_use
func @create_and_use(%start: i64, %end: i64) {
  // CHECK: %[[DS:.*]] = rd.range
  %ds = rd.range %start to %end : (i64, i64) -> !rd.Dataset
  // CHECK: %[[ITR:.*]] = rd.make_iterator %[[DS]]
  %itr = rd.make_iterator %ds : (!rd.Dataset) -> !rd.Iterator
  // CHECK: %[[IS_VALID:.*]], %[[VALUE:.*]] = rd.iterator_next %[[ITR]]
  %valid, %value = rd.iterator_next %itr : (!rd.Iterator) -> (i1, i64)
  return
}
