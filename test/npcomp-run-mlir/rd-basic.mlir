// RUN: npcomp-run-mlir %s -invoke basic -arg-value="dense<[1.0]> : tensor<1xf32>" -shared-libs=%npcomp_runtime_shlib 2>&1 | FileCheck %s

llvm.func @printf(!llvm.ptr<i8>, ...) -> !llvm.i32
llvm.mlir.global internal constant @foo("here %d\n\00")

func @basic2(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = llvm.mlir.addressof @foo : !llvm.ptr<array<9 x i8>>
  %1 = llvm.mlir.constant(0: i32) : !llvm.i32
  %const = llvm.mlir.constant(32: i32) : !llvm.i32
  %2 = llvm.getelementptr %0[%1, %1] : (!llvm.ptr<array<9 x i8>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i8>
  // CHECK: here
  %3 = llvm.call @printf(%2, %const) : (!llvm.ptr<i8>, !llvm.i32) -> !llvm.i32
  %4 =  tcf.add %arg0, %arg0 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %4 : tensor<?xf32>
}


func @basic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // TODO: add a step to the range definition.
  %0 = llvm.mlir.constant(0 : i64) : !llvm.i64
  %1 = llvm.mlir.constant(3 : i64) : !llvm.i64
  %2 = llvm.mlir.undef : !llvm.struct<"range_state", (i64, i64, i64)> // (curr, end, step)
  %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<"range_state", (i64, i64, i64)>
  %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<"range_state", (i64, i64, i64)>
  %5 = llvm.insertvalue %0, %4[2] : !llvm.struct<"range_state", (i64, i64, i64)>  // TODO: use me as step!

  %str0 = llvm.mlir.addressof @foo : !llvm.ptr<array<9 x i8>>
  %str1 = llvm.mlir.constant(0: i32) : !llvm.i32
  %str = llvm.getelementptr %str0[%str1, %str1] : (!llvm.ptr<array<9 x i8>>, !llvm.i32, !llvm.i32) -> !llvm.ptr<i8>
  br ^bb1(%5 : !llvm.struct<"range_state", (i64, i64, i64)>)

^bb1(%itr: !llvm.struct<"range_state", (i64, i64, i64)>):
  %toprint = llvm.extractvalue %itr[0] : !llvm.struct<"range_state", (i64, i64, i64)>
  %end = llvm.extractvalue %itr[1] : !llvm.struct<"range_state", (i64, i64, i64)>
//  %toprint.casted = llvm.bitcast %toprint : !i64 to !llvm.i64
  %unused = llvm.call @printf(%str, %toprint) : (!llvm.ptr<i8>, !llvm.i64) -> !llvm.i32
  %increment = llvm.mlir.constant(1 : i64) : !llvm.i64  // TODO: use increment field!
  %next = llvm.add %toprint, %increment : !llvm.i64
  %next.state = llvm.insertvalue %next, %itr[0] : !llvm.struct<"range_state", (i64, i64, i64)>
  %eqr = llvm.icmp "ne" %next, %end : !llvm.i64
  llvm.cond_br %eqr, ^bb1(%next.state : !llvm.struct<"range_state", (i64, i64, i64)>), ^bb2
^bb2:
  return %arg0 : tensor<?xf32>
}