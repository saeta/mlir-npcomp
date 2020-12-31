//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RD_OPS
#define RD_OPS

include "npcomp/Dialect/RD/IR/RDBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class RD_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<RD_Dialect, mnemonic, traits> {
}

class RD_DatasetOp<string mnemonic>
    : RD_Op<mnemonic, [NoSideEffect]> {
}

def RD_RangeOp : RD_DatasetOp<"range"> {
    let summary = "Range of values.";
    let description = [{
        A sequence of consecutive integers.
    }];

    let arguments = (ins I64:$start, I64:$end);
    let results = (outs RD_Dataset:$result);

    let assemblyFormat = "$start `to` $end attr-dict `:` functional-type(operands, results)";
}

// TODO: Consider marking as no-side-effect.
def RD_MakeIteratorOp : RD_Op<"make_iterator"> {
    let summary = "Build an iterator for a dataset.";
    let description = [{
        Returns an !rd.Iterator type that corresponds to this logical dataset.
    }];

    let arguments = (ins RD_Dataset:$dataset);
    let results = (outs RD_Iterator:$result);

    let assemblyFormat = "$dataset attr-dict `:` functional-type(operands, results)";  // TODO: just write the element type in future?
}

def RD_IteratorNextOp : RD_Op<"iterator_next"> {
    let summary = "Advances the iterator by one and returns the current values.";
    let description = [{
        Returns a tuple of values:
         1. A Boolean indicating if there was a valid value.
         2. The value (only valid if the boolean is true).
    }];

    let arguments = (ins RD_Iterator:$iterator);
    let results = (outs I1:$valid, I64:$value);  // TODO: make generic over values.

    let assemblyFormat = "$iterator attr-dict `:` functional-type(operands, results)";  // TODO: just write the element type in future?
}

def RD_PrintOp : RD_Op<"print"> {
    let summary = "Print a value for debugging purposes.";

    let arguments = (ins I64:$value);
}

#endif // #ifndef RD_OPS
