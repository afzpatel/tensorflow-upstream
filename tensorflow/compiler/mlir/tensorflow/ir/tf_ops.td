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

// This is the operation definition file for TensorFlow.
//
// This file contains TensorFlow ops whose definitions are amended to fix
// issues or provide more information. In this file you have full control
// of the op definition; all changes will be retained with subsequent
// refreshes.
//
// This file includes another file, `tf_generated_ops.td`, which contains
// all ops whose definitions are generated from TensorFlow codebase.
// Changes made there are not respected.

#ifdef TF_OPS
#else
#define TF_OPS

include "tensorflow/compiler/mlir/tensorflow/ir/tf_generated_ops.td"

class TF_TensorListInitOp<string mnemonic> : TF_Op<mnemonic, [NoMemoryEffect]> {
  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeAttr shape_type = TF_DerivedOperandTypeAttr<0>;

//  let verifier = [{
//    if (handle_dtype().getSubtypes().size() != 1) {
//      return emitOpError(
//          "must have exactly one subtype in the result variant type");
//    }
//
//    return Verify(*this);
//  }];

  DerivedTypeAttr element_dtype = DerivedTypeAttr<
      "return getElementTypeOrSelf(element_type());">;

  let extraClassDeclaration = [{
    // Returns type of the TensorList element produced by this op.
    TensorType element_type() { return handle_dtype().getSubtypes()[0]; }

    // Returns data type of the result handle. Returned type contains type of
    // the TensorList element as a subtype.
    VariantType handle_dtype() {
      return getElementTypeOrSelf(handle()->getType()).cast<TF::VariantType>();
    }
  }];
}

// In MLIR, the TensorFlow tensor value is represented as an ElementsAttr, with
// its type encoding the tensor's shape and data type.
def TF_ConstOp : TF_Op<"Const", [NoMemoryEffect]> {
  let summary = "Constant tensor op";

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;

  let builders = [
    OpBuilder<(ins "Attribute":$value)>,
    OpBuilder<(ins "Type":$type, "Attribute":$value)>,
  ];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return BroadcastCompatible(l, r);
    }
  }];
}

def TF_EmptyTensorListOp : TF_TensorListInitOp<"EmptyTensorList"> {
  let summary = "Creates and returns an empty tensor list.";

  let description = [{
All list elements must be tensors of dtype element_dtype and shape compatible
with element_shape.

handle: an empty tensor list.
element_dtype: the type of elements in the list.
element_shape: a shape compatible with that of elements in the list.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$element_shape,
    I32Tensor:$max_num_elements
  );
}

// TODO(fengliuai): The tf.Identity is side-effect free and it doesn't change
// the status of the system during the execution. However it shouldn't be folded
// in general if it used to serve for caching and some other invariant checks,
// so we removed the side-effect free property in the op definition. This is a
// hack, and we should fix it if we have a better way to model it.
def TF_IdentityOp : TF_Op<"Identity", [TF_OperandsSameAsResultsTypeOrRef]> {
  let summary = "Identity op";

  let description = [{
Returns a tensor with the same shape and contents as input.
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_IfOp : TF_Op<"If", []> {
  let summary = "output = cond ? then_branch(input) : else_branch(input)";

  let description = [{
output = cond ? then_branch(input) : else_branch(input)

cond: A Tensor. If the tensor is a scalar of non-boolean type, the
    scalar is converted to a boolean according to the
    following rule: if the scalar is a numerical value, non-zero means
    True and zero means False; if the scalar is a string, non-empty
    means True and empty means False. If the tensor is not a scalar,
    being empty means False and being non-empty means True.
input: A list of input tensors.
then_branch: A function that takes 'inputs' and returns a list of
    tensors, whose types are the same as what else_branch returns.
else_branch: A function that takes 'inputs' and returns a list of
    tensors.  whose types are the same as what then_branch returns.
  }];

  let arguments = (ins
    TF_Tensor:$cond,
    Variadic<TF_Tensor>:$input,

    FlatSymbolRefAttr:$then_branch,
    FlatSymbolRefAttr:$else_branch,

    // Used to map StatelessIf and If op defined in TensorFlow to a common op.
    BoolAttr:$is_stateless
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeAttr Tcond = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<1>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;
  //TF_DerivedResultShapeListAttr output_shapes = TF_DerivedResultShapeListAttr<0>;

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    // Resolve the then branch function.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    func::FuncOp ResolveThenFunction(::mlir::SymbolTableCollection* table) {
      if (table)
        return table->lookupNearestSymbolFrom<func::FuncOp>(*this, getThenBranchAttr());
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        *this, getThenBranchAttr());
    }
    // TODO(b/204997177): Deprecate and remove.
    func::FuncOp then_function(::mlir::SymbolTableCollection* table = nullptr) {
      return ResolveThenFunction(table);
    }

    // Resolve the else branch function.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    func::FuncOp ResolveElseFunction(::mlir::SymbolTableCollection* table) {
      if (table)
        return table->lookupNearestSymbolFrom<func::FuncOp>(*this, getElseBranchAttr());
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        *this, getElseBranchAttr());
    }
    // TODO(b/204997177): Deprecate and remove.
    func::FuncOp else_function(::mlir::SymbolTableCollection* table = nullptr) {
      return ResolveElseFunction(table);
    }
  }];
}

def TF_MeanOp : TF_Op<"Mean", [NoMemoryEffect]> {
  let summary = "Computes the mean of elements across dimensions of a tensor.";

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    TF_NumberTensor:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TF_NumberTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

// In MLIR, the 'tf.Placeholder.input' instruction is used to capture attributes
// of function arguments.
// Note: NoMemoryEffect trait is not added intentionally to preserve the captured
// attributes even if the input is unused.
def TF_PlaceholderInputOp : TF_Op<"Placeholder.input",
    [SameOperandsAndResultType]> {
  let summary = "PlaceholderInput op";

  let description = [{
Inserts a placeholder for a tensor that will be always fed.
  }];

  let arguments = (ins
    TF_Tensor:$arg,

    OptionalAttr<F32Attr>:$min,
    OptionalAttr<F32Attr>:$max,
    OptionalAttr<TF_IntTypeAttr>:$type
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_PlaceholderOp : TF_Op<"Placeholder", [NoMemoryEffect]> {
  let summary = "Placeholder op";

  let description = [{
Inserts a placeholder for a tensor that will be always fed.
  }];

  let arguments = (ins
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_WhileOp : TF_Op<"While", []> {
  let summary = [{
output = input; While (Cond(output)) { output = Body(output) }
  }];

  let description = [{
output = input; While (Cond(output)) { output = Body(output) }

input: A list of input tensors whose types are T.
output: A list of output tensors whose types are T.
cond: A function that takes 'input' and returns a tensor.  If the tensor is
    a scalar of non-boolean, the scalar is converted to a boolean
    according to the following rule: if the scalar is a numerical
    value, non-zero means True and zero means False; if the scalar is
    a string, non-empty means True and empty means False. If the
    tensor is not a scalar, non-emptiness means True and False
    otherwise.
body: A function that takes a list of tensors and returns another
      list of tensors. Both lists have the same types as specified
      by T.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$input,

    FlatSymbolRefAttr:$cond,
    FlatSymbolRefAttr:$body,
    ConfinedAttr<DefaultValuedOptionalAttr<I64Attr, "10">, [IntMinValue<1>]>:$parallel_iterations,

    // Used to map StatelessWhile and While op defined in TensorFlow to a common
    // op.
    BoolAttr:$is_stateless,

    // In TensorFlow, While has a special behavior where if `output_shapes`
    // attribute is not empty, those shapes are used in its shape function
    // as result shapes instead of propagating operand shapes as result shapes.
    // This allows for different result shapes from operand shapes. While these
    // shapes are imported and set as a part of the result type, there is no
    // indicator differentiating between having no output shapes compared to
    // having all unranked shapes. Thus this attribute is set to determine
    // which shape function behavior to use for this op, specifically
    // propagating operand shapes as result shapes when this attribute is not
    // set, or preserving result shapes as is when this attribute is set.
    UnitAttr:$shape_invariant
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;
  //TF_DerivedResultShapeListAttr output_shapes = TF_DerivedResultShapeListAttr<0>;

  let extraClassDeclaration = [{
    // Get the condition function.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    func::FuncOp ResolveCondFunction(::mlir::SymbolTableCollection* table) {
      if (table)
        return table->lookupNearestSymbolFrom<func::FuncOp>(*this, getCondAttr());
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getCondAttr());
    }
    // TODO(b/204997177): Deprecate and remove.
    func::FuncOp cond_function() { return ResolveCondFunction(nullptr); }

    // Get the body function.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    func::FuncOp ResolveBodyFunction(::mlir::SymbolTableCollection* table) {
      if (table)
        return table->lookupNearestSymbolFrom<func::FuncOp>(*this, getBodyAttr());
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getBodyAttr());
    }
    // TODO(b/204997177): Deprecate and remove.
    func::FuncOp body_function() { return ResolveBodyFunction(nullptr); }
  }];
}

def TF_TensorListReserveOp : TF_TensorListInitOp<"TensorListReserve"> {
  let summary = "List of the given size with empty elements.";

  let description = [{
element_shape: the shape of the future elements of the list
num_elements: the number of elements to reserve
handle: the output list
element_dtype: the desired type of elements in the list.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$element_shape,
    I32Tensor:$num_elements
  );
}

#endif // TF_OPS
