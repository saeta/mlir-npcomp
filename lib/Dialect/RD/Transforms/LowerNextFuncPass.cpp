//===- BuildInitFunc.cpp - Extracts a pipeline definition ---*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <set>

#include "PassDetail.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/RD/IR/RDDatasetInterface.h"
#include "npcomp/Dialect/RD/IR/RDDialect.h"
#include "npcomp/Dialect/RD/IR/RDOps.h"
#include "npcomp/Dialect/RD/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::NPCOMP;

#define DEBUG_TYPE "rd-lower-next-func"

namespace {

class ConvertIteratorIndex : public OpRewritePattern<rd::IteratorIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(rd::IteratorIndexOp srcOp, PatternRewriter &rewriter) const override {
    if (!srcOp.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(srcOp, "assumes only a single user");
    }
    auto user = *srcOp.getResult().user_begin();
    if (rd::DatasetNext datasetOp = dyn_cast<rd::DatasetNext>(user)) {
      auto iteratorTyOpt = datasetOp.buildStateLLVMType();
      if (iteratorTyOpt) {
        auto iteratorTy = *iteratorTyOpt;
        auto* context = srcOp.getContext();
        auto int64Ty = LLVM::LLVMType::getInt64Ty(context);

        llvm::dbgs() << "HERE!!!!\n";

        auto zero = rewriter.create<LLVM::ConstantOp>(srcOp.getLoc(), int64Ty,
          rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
        llvm::dbgs() << "zero: " << zero << "\n";
        auto indexAttr = srcOp.childIndex();
        llvm::dbgs() << "indexAttr: " << indexAttr << "\n";
        auto attrRange = indexAttr.getAsRange<IntegerAttr>();
        auto index = rewriter.create<LLVM::ConstantOp>(srcOp.getLoc(), int64Ty, *attrRange.begin());
        llvm::dbgs() << "index: " << index << "\n";
        rewriter.replaceOpWithNewOp<LLVM::GEPOp>(srcOp, iteratorTy, srcOp.parent(), ValueRange({zero, index}));
        llvm::dbgs() << "here: " << srcOp << "\n";
        llvm::dbgs() << *rewriter.getInsertionBlock()->getParentOp() << "\n";
        return success();
      } else {
        rewriter.replaceOp(srcOp, {srcOp.parent()});  // Forward parent ptr along.
        return success();
      }
    } else {
      return rewriter.notifyMatchFailure(srcOp, "single user not a DatasetNext op");
    }
  }
};

// Clones the definition function, transforming the ops used to the `[...].next` variations of the ops.
class LowerNextFunc : public RDLowerNextFuncBase<LowerNextFunc> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto* context = &getContext();
    auto pipelineDefOp = getOperation();

    OwningRewritePatternList patterns;
    patterns.insert<ConvertIteratorIndex>(context);
    (void)applyPatternsAndFoldGreedily(pipelineDefOp, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<OperationPass<rd::PipelineDefinitionOp>> mlir::NPCOMP::createLowerNextFuncPass() {
  return std::make_unique<LowerNextFunc>();
}
