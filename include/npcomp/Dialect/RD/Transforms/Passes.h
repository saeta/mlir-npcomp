//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_RD_TRANSFORMS_PASSES_H
#define NPCOMP_DIALECT_RD_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace NPCOMP {

std::unique_ptr<OperationPass<FuncOp>> createRDMergeFuncsPass();
std::unique_ptr<OperationPass<ModuleOp>> createRDLowerToLLVMPass();

/// Registers all RD transformation passes.
void registerRDPasses();

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_RD_TRANSFORMS_PASSES_H
