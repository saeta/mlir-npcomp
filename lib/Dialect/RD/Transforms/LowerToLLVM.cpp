//===- MergeFuncs.cpp - Bufferization for TCP dialect -------------*-
//C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/RD/IR/RDDialect.h"
#include "npcomp/Dialect/RD/IR/RDOps.h"
#include "npcomp/Dialect/RD/Transforms/Passes.h"
#include "npcomp/Dialect/Refback/IR/RefbackDialect.h"
#include "npcomp/Dialect/Refback/IR/RefbackOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {

// class MonolithicRDLoweringPass
//     : public PassWrapper<MonolithicRDLoweringPass, OperationPass<ModuleOp>> {
//   MonolithicRDLoweringPass() = default;
// public:
//   void getDependentDialects(::mlir::DialectRegistry &registry) const override
//   {
//     registry.insert<LLVM::LLVMDialect>();
//   }
//   void runOnOperation() override {
//     ModuleOp m = getOperation();
//     m.walk([&](rd::MakeIteratorOp op) {

//       llvm::outs() << "\nHello!!" << op << "\n\n";
//     });
//     // TODO: HERE!
//   }
// };

constexpr char kCreatePrefix[] = "__rd_create_";
constexpr char kNextPrefix[] = "__rd_next_";

// TODO: Add a new region op type to allow for parallelized compilation! (This
// is too monolithic!)
class LowerToRuntimePass : public LowerToRuntimeBase<LowerToRuntimePass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    // TODO: Remove these!
    registry.insert<scf::SCFDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();
    OpBuilder builder(module);

    // TODO: mapping from MakeIteratorOp to struct types.
    // TODO: investigate faster ADTs.
    std::map<rd::MakeIteratorOp, LLVM::LLVMType> states;
    module.walk([&](rd::MakeIteratorOp op) {

      llvm::outs() << "\nHello!!\n" << op << "\n";
      llvm::outs() << op.getOperand() << "\n\n";

      SmallVector<LLVM::LLVMType, 6> state_fields;
      bool shouldContinue = true;
      // TODO: De-dupe code-gen based on the defining op!
      auto walkOp = op.getOperand().getDefiningOp();
      // TODO: Convert this to handle DAGs of iteration!
      while (shouldContinue) {
        llvm::outs() << "Got a dialect for op " << *walkOp << ": " << walkOp->getName().getDialect() << "\n";
        llvm::outs() << "walkOp name stringref: '" << walkOp->getName().getStringRef() << "'\n";
        if (walkOp->getName().getDialect() != "rd") {
          llvm_unreachable("Unexpected input operation when building state datastructure.");
        }
        if (walkOp->getName().getStringRef() == "rd.range") {
          auto int64Ty = LLVM::LLVMType::getInt64Ty(context);
          auto boolTy = LLVM::LLVMType::getInt1Ty(context);
          auto stateTy =
              LLVM::LLVMType::getStructTy(context, {int64Ty, int64Ty/*, int64Ty (TODO: insert step)*/});
          auto statePtrTy = stateTy.getPointerTo();
          state_fields.push_back(stateTy);

          // Make init function.
          auto createName = (Twine(kCreatePrefix) + "foo_fix_me").str();

          auto createFnTy = LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getVoidTy(context), {statePtrTy}, /*isVarArg=*/false);
          auto createFn = builder.create<LLVM::LLVMFuncOp>(
            op.getLoc(), createName, createFnTy, LLVM::Linkage::Internal);
          {
            Block *createBody = createFn.addEntryBlock();
            auto createBuilder = OpBuilder::atBlockBegin(createBody);
            auto statePtr = createBody->getArgument(0);
            auto zero = createBuilder.create<LLVM::ConstantOp>(op.getLoc(), int64Ty, builder.getIntegerAttr(builder.getIndexType(), 0));
            auto one = createBuilder.create<LLVM::ConstantOp>(op.getLoc(), int64Ty, builder.getIntegerAttr(builder.getIndexType(), 1));
            auto startPtr = createBuilder.create<LLVM::GEPOp>(op.getLoc(), statePtrTy, statePtr, ValueRange({zero, zero}));
            auto endPtr = createBuilder.create<LLVM::GEPOp>(op.getLoc(), statePtrTy, statePtr, ValueRange({zero, one}));
            auto startValueOp = createBuilder.clone(*walkOp->getOperand(0).getDefiningOp());
            auto endValueOp = createBuilder.clone(*walkOp->getOperand(1).getDefiningOp());
            createBuilder.create<LLVM::StoreOp>(op.getLoc(), startPtr, startValueOp->getResult(0));
            createBuilder.create<LLVM::StoreOp>(op.getLoc(), endPtr, endValueOp->getResult(0));

            // auto startInserted = createBuilder.create<LLVM::InsertValueOp>(op.getLoc(), undefStruct, walkOp->getOperand(0), builder.getI64ArrayAttr(0));
            // auto endInserted = createBuilder.create<LLVM::InsertValueOp>(op.getLoc(), startInserted, walkOp->getOperand(1), builder.getI64ArrayAttr(1));
            // createBuilder.create<LLVM::ReturnOp>(op.getLoc(), endInserted.getResult());
            createBuilder.create<ReturnOp>(op.getLoc());
            llvm::outs() << "Made a create fn:\n" << createFn << "\n";
          }

          // Make next function.
          auto nextName = (Twine(kNextPrefix) + "foo_fix_me").str();

          auto nextReturnTy = LLVM::LLVMType::getStructTy(context, {boolTy, int64Ty});
          // TODO: set attributes on statePtrTy here & above: noalias nocapture [sret] dereferenceable(##)
          auto nextFnTy = LLVM::LLVMType::getFunctionTy(nextReturnTy, {statePtrTy}, /*isVarArg=*/false);
          auto nextFn = builder.create<LLVM::LLVMFuncOp>(op.getLoc(), nextName, nextFnTy, LLVM::Linkage::Internal);

          {
            auto *nextBody = nextFn.addEntryBlock();
            auto nextBuilder = OpBuilder::atBlockBegin(nextBody);
            auto inputStateArgPtr = nextBody->getArgument(0);
            auto zero = nextBuilder.create<LLVM::ConstantOp>(op.getLoc(), int64Ty, builder.getIntegerAttr(builder.getIndexType(), 0));
            auto one = nextBuilder.create<LLVM::ConstantOp>(op.getLoc(), int64Ty, builder.getIntegerAttr(builder.getIndexType(), 1));
            auto resultPtr = nextBuilder.create<LLVM::GEPOp>(op.getLoc(), statePtrTy, inputStateArgPtr, ValueRange({zero, zero}));
            auto maxPtr = nextBuilder.create<LLVM::GEPOp>(op.getLoc(), statePtrTy, inputStateArgPtr, ValueRange({zero, one}));
            auto result = nextBuilder.create<LLVM::LoadOp>(op.getLoc(), resultPtr);
            auto max = nextBuilder.create<LLVM::LoadOp>(op.getLoc(), maxPtr);
            auto nextResult = nextBuilder.create<LLVM::AddOp>(op.getLoc(), result, one);
            auto isValid = nextBuilder.create<LLVM::ICmpOp>(op.getLoc(), LLVM::ICmpPredicate::ne, result, max);
            nextBuilder.create<LLVM::StoreOp>(op.getLoc(), resultPtr, nextResult);
            nextBuilder.create<ReturnOp>(op.getLoc(), ValueRange({isValid, nextResult}));
            llvm::outs() << "Made a next function:\n" << nextFn << "\n";
          }
          // Replace ops!
          // walkOp->erase();
          walkOp->remove();  // TODO: Erase?
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfter(op);
            auto one = builder.create<LLVM::ConstantOp>(op.getLoc(), int64Ty, builder.getIntegerAttr(builder.getIndexType(), 1));
            auto itr = builder.create<LLVM::AllocaOp>(op.getLoc(), statePtrTy, ValueRange({one}));
            builder.create<LLVM::CallOp>(op.getLoc(), createFn, ValueRange({itr}));
            op.result().replaceAllUsesWith(itr);
            // op.erase();
            op->remove();  // TODO: Erase later!
            llvm::outs() << "Did some sugary! Things now look like:\n"
                         << module << "\n";

            for (auto i = itr.getResult().user_begin(); i != itr.getResult().user_end(); ++i) {
              llvm::outs() << "Walking users.... found: " << i->getName() << "... ";
              if (i->getName().getStringRef() == "rd.iterator_next") {
                llvm::outs() << "MATCHING!\n";
                OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointAfter(*i);
                auto nextCallOp = builder.create<LLVM::CallOp>(op.getLoc(), nextFn, ValueRange({itr}));
                auto isValid = builder.create<LLVM::ExtractValueOp>(op.getLoc(), nextReturnTy, nextCallOp.getResult(0), builder.getI32ArrayAttr(0));
                auto nextValue = builder.create<LLVM::ExtractValueOp>(op.getLoc(), nextReturnTy, nextCallOp.getResult(0), builder.getI32ArrayAttr(1));
                i->getResult(0).replaceAllUsesWith(isValid);
                i->getResult(1).replaceAllUsesWith(nextValue);
                i->remove();  // TODO: erase?
              } else {
                llvm::outs() << "didn't match.\n";
              }
            }
            llvm::outs() << "Did some more sugary! Things now look like:\n" << module << "\n";
          }

          // walkOp->getResult(0).get
          shouldContinue = false;
        } else {
          llvm_unreachable("Unexpected dataset operation.");
        }
      }
    });

    LLVMTypeConverter converter(context);

    OwningRewritePatternList patterns;
    LLVMConversionTarget target(*context);
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });

    populateStdToLLVMConversionPatterns(converter, patterns);
    // TODO: populate patterns.

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
    // TODO: Implement me!
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::createRDLowerToLLVMPass() {
  return std::make_unique<LowerToRuntimePass>();
}
