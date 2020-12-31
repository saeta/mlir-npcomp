// RUN: npcomp-opt <%s -rd-lower-to-llvm | FileCheck %s

func @main() {
  %start = constant 1 : i64
  %end = constant 3 : i64
  %dataset = rd.range %start to %end : (i64, i64) -> !rd.Dataset
  %itr = rd.make_iterator %dataset : (!rd.Dataset) -> !rd.Iterator
  %hn0, %val0 = rd.iterator_next %itr : (!rd.Iterator) -> (i1, i64)
  "rd.print"(%val0) : (i64) -> ()
  return
}
// CHECK-LABEL: func @__rd_make_ds
// CHECK: 