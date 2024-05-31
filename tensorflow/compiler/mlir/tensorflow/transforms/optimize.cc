/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <iostream>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {
namespace {

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_optimize.inc"

// Canonicalize operations in functions.
struct TFOptimizePass : public FunctionPass<TFOptimizePass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto func = getFunction();
    populateWithGenerated(&getContext(), &patterns);
    applyPatternsGreedily(func, patterns);
  }
};

}  // namespace

std::unique_ptr<FunctionPassBase> CreateTFOptimizePass() {
  return std::make_unique<TFOptimizePass>();
}

static PassRegistration<TFOptimizePass> pass("tf-optimize", "Optimizes TF.");

}  // namespace TF
}  // namespace mlir
