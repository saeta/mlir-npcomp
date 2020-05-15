//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TCP_OPS
#define TCP_OPS

include "npcomp/Dialect/TCP/IR/TCPBase.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/SideEffects.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class TCP_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<TCP_Dialect, mnemonic, traits> {
}

// TODO: clarify allowed tensor element types.
// TODO: HasParent is too restrictive? can't have an island with loop.for with
// further ops inside it?
def TCP_AddOp
    : TCP_Op<"add", []> {
  let summary = "Adds two tensors.";
  let description = [{
Adds two tensors.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
}

def TCP_BroadcastToOp : TCP_Op<"broadcast_to"> {
  let summary = "Broadcasts an operand to a given shape.";
  let description = [{
Broadcasts `operand` to the shape `shape`.

It is undefined behavior if such a broadcast is not legal.
  }];
  let arguments = (ins AnyRankedTensor:$operand, Shape_ShapeType:$shape);
  let results = (outs AnyRankedTensor:$result);
}

//===----------------------------------------------------------------------===//
// Ops that need to be factored to a proper home.
//===----------------------------------------------------------------------===//
// TODO: Find a home for these.

// TODO: This probably doesn't belong in the tcp dialect.
def TCP_AllocMemRefOp : TCP_Op<"alloc_memref", []> {
  let summary = "Allocates a memref of the given shape.";
  let description = [{
Allocates a memref of the given shape.
  }];
  let arguments = (ins Shape_ShapeType:$shape);
  let results = (outs AnyMemRef:$memref);
  let assemblyFormat = "$shape attr-dict `:`  type($memref)";
}

// TODO: Change to a more principled error handling mechanism.
// This op probably doesn't need to exist eventually.
// This op is also not correctly modeled right now, since it itself doesn't
// produce the error in practice. The ops like shape.broadcast itself, when
// lowered, immediately produce errors.
// Right now, it's more of an "observe_error" which just keeps NoSideEffect
// shape ops alive.
def TCP_AbortIfErrorOp : TCP_Op<"abort_if_error",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Aborts the program if the argument is an error shape.";
  let description = [{
Aborts the program if its `shape` argument is an error shape.
  }];
  let arguments = (ins Shape_ShapeType:$shape);
  // TODO: ODS seems to create redeclared class members if we remove this,
  // resulting in C++ compilation errors.
  let results = (outs NoneType:$dummy);
}

// TODO: This probably belongs in the shape dialect.
def TCP_GetExtentOp : TCP_Op<"get_extent",
    [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Gets the specified extent from a shape.";
  let description = [{
Gets the specified extent from a shape.

This op has undefined behavior if the shape is an error.
  }];
  let arguments = (ins Shape_ShapeType:$shape, I64Attr:$dim);
  let results = (outs Index:$extent);
  let assemblyFormat = "$shape `,` $dim attr-dict";
}

// TODO: Move this op to a more comprehensive "npcomp_rt" or "npcomp_hal"
// dialect that properly demarcates the low-level interface to the npcomp
// runtime.
//
// Note: This op operates on tensors since memref is not general enough to
// represent the runtime tensor representations that we need in npcomp. In
// fact, we may want it to operate on a proper runtime-specific type
// instead of a "tensor".
def TCP_RtGetTensorExtentOp : TCP_Op<"rt_get_tensor_extent",
    [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Gets the extent of a tensor in a runtime specific way";
  let description = [{
    This op interfaces with an external runtime to obtain the extent of a
    tensor's dimension.
  }];
  let arguments = (ins AnyRankedTensor:$tensor, I64Attr:$dim);
  let results = (outs Index:$extent);
}

// TODO: This op belongs in the shape dialect as `shape.from_extents`.
def TCP_ShapeFromExtentsOp : TCP_Op<"shape_from_extents",
    [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Constructs a shape from extents";
  let description = [{
    Constructs a shape from the extents passed as arguments.
  }];
  let arguments = (ins Variadic<Index>:$extents);
  let results = (outs Shape_ShapeType:$shape);
}

def TCP_AbortIfOp : TCP_Op<"abort_if"> {
  let summary = "Aborts the program if the argument is true.";
  let description = [{
  Aborts the program if the argument is true.

  TODO: Support a custom message.
  }];
  let arguments = (ins I1:$pred);
  let results = (outs);
}

#endif // TCP_OPS