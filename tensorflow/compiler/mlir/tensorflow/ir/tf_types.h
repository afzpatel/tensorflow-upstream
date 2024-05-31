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

// This file defines the types used in the standard MLIR TensorFlow dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TYPES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TYPES_H_

#if 1

// 1.x version, which does not compile, because the concept of Type::Kind is no longer present in MLIR

namespace mlir {
namespace TF {

namespace TensorFlowTypes {
// List of supported TensorFlowType kinds, necessary for isa/dyn_cast.

enum Kind {
  FIRST_USED_TENSORFLOW_TYPE,
#define HANDLE_TF_TYPE(tftype, enumerant, name) enumerant,
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
  LAST_USED_TENSORFLOW_TYPE = FIRST_USED_TENSORFLOW_TYPE + 0xff,
};
}  // namespace TensorFlowTypes

// The base class in the tensor flow type hierarchy.
class TensorFlowType : public Type {
 public:
  using Type::Type;
#if 0
  // Support method to enable LLVM-style type casting.
  static bool classof(Type type) {
    return type.getKind() >= Type::FIRST_TENSORFLOW_TYPE &&
           type.getKind() <= TensorFlowTypes::LAST_USED_TENSORFLOW_TYPE;
  }
#endif  
};

// Returns true if the specified type is a valid TensorFlow element type.
static inline bool IsValidTFElementType(Type type) {
  return type.isa<FloatType>() || type.isa<IntegerType>() ||
         type.isa<TensorFlowType>();
}

// Returns true if this is a valid TensorFlow tensor type.
static inline bool IsValidTFTensorType(Type type) {
  // TensorFlow types should be tensors of one of the valid TensorFlow element
  // types.
  if (auto tensor_ty = type.dyn_cast<TensorType>())
    return IsValidTFElementType(tensor_ty.getElementType());
  return false;
}

namespace detail {
// Common implementation of TensorFlow types.  The template argument indicates
// the concrete derived class per CRTP.  Concrete classes must implement the
// following:
//   - `static unsigned getTypeKind()` that returns the (fixed) kind of the
//     type.
template <typename Derived>
class TensorFlowTypeImpl : public Type::TypeBase<Derived, TensorFlowType, TypeStorage> {
 public:
  using Base = typename Type::TypeBase<Derived, TensorFlowType, TypeStorage>;
  using TFBase = TensorFlowTypeImpl<Derived>;
  using Base::Base;

  // Get the unique'ed type in the given context.
  static Derived get(MLIRContext *context) {
    return Base::get(context, Derived::getTypeKind());
  }

  // Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == Derived::getTypeKind(); }
};
}  // namespace detail

#define HANDLE_TF_TYPE(tftype, enumerant, name)                          \
  class tftype##Type : public detail::TensorFlowTypeImpl<tftype##Type> { \
   public:                                                               \
    using TFBase::TFBase;                                                \
    static unsigned getTypeKind() { return TensorFlowTypes::enumerant; } \
  };

// Custom TensorFlow types are defined separately.
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)

// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"

// Storage type contains inferred subtypes for VariantType.
class VariantTypeStorage : public TypeStorage {
 public:
  using KeyTy = ArrayRef<TensorType>;

  // NOLINTNEXTLINE
  static VariantTypeStorage* construct(TypeStorageAllocator& allocator,
                                       const KeyTy& key) {
    ArrayRef<TensorType> subtypes = allocator.copyInto(key);
    return new (allocator.allocate<VariantTypeStorage>())
        VariantTypeStorage(subtypes);
  }

  explicit VariantTypeStorage(const KeyTy& key) : subtypes_(key) {}

  bool operator==(const KeyTy& key) const { return key == subtypes_; }

  static llvm::hash_code hashKey(const KeyTy& key) {
    return llvm::hash_combine_range(key.begin(), key.end());
  }

  KeyTy subtypes_;
};

// TensorFlow variant type is used to support arbitrary custom C++ data types.
// VariantType stores inferred shape and datatype for subtypes unlike most other
// data types don't have any associated information. These subtypes are opaque
// and their interpretation depends on the actual underlying type. For example,
// variants encoding TensorList type stores the common shape and dtype of the
// list elements as the only subtype.
class VariantType
    : public Type::TypeBase<VariantType, TensorFlowType, VariantTypeStorage> {
 public:
  using Base::Base;

  static VariantType get(ArrayRef<TensorType> subtypes, MLIRContext* context) {
    return Base::get(context, TensorFlowTypes::VARIANT, subtypes);
  }

  static VariantType getChecked(ArrayRef<TensorType> subtypes,
                                MLIRContext* context, Location loc) {
    return Base::getChecked(loc, context, TensorFlowTypes::VARIANT, subtypes);
  }

  static VariantType get(MLIRContext* context) { return get({}, context); }

  static bool kindof(unsigned kind) { return kind == TensorFlowTypes::VARIANT; }

  static LogicalResult verifyConstructionInvariants(
      std::optional<Location> loc, MLIRContext* context,
      ArrayRef<TensorType> subtypes) {
    // Each of the subtypes should be a valid TensorFlow type.
    for (TensorType subtype : subtypes) {
      if (!IsValidTFTensorType(subtype)) {
        if (loc) {
          emitError(*loc) << "invalid VariantType subtype: " << subtype;
        }
        return failure();
      }
    }
    return success();
  }

  ArrayRef<TensorType> getSubtypes() { return getImpl()->subtypes_; }
};

}  // end namespace TF
}  // end namespace mlir

#else

// 2.x version, depends on tensorflow/core/ir/... (which is absent from 1.x)

#include "tensorflow/core/ir/types/dialect.h"

namespace mlir {
namespace TF {

// This all moved under tensorflow/core/ir/types and these using declaration are
// to help with the transition.

using ::mlir::tf_type::AreCastCompatible;          // NOLINT
using ::mlir::tf_type::ArraysAreCastCompatible;    // NOLINT
using ::mlir::tf_type::BroadcastCompatible;        // NOLINT
using ::mlir::tf_type::DropRefType;                // NOLINT
using ::mlir::tf_type::filter_resources;           // NOLINT
using ::mlir::tf_type::GetCastCompatibleType;      // NOLINT
using ::mlir::tf_type::HasCompatibleElementTypes;  // NOLINT
using ::mlir::tf_type::IsValidTFTensorType;        // NOLINT
using ::mlir::tf_type::OperandShapeIterator;       // NOLINT
using ::mlir::tf_type::ResourceType;               // NOLINT
using ::mlir::tf_type::ResultShapeIterator;        // NOLINT
using ::mlir::tf_type::ResultShapeRange;           // NOLINT
using ::mlir::tf_type::StringType;                 // NOLINT
using ::mlir::tf_type::TensorFlowRefType;          // NOLINT
using ::mlir::tf_type::TensorFlowType;             // NOLINT
using ::mlir::tf_type::TensorFlowTypeWithSubtype;  // NOLINT
using ::mlir::tf_type::VariantType;                // NOLINT

#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  using tftype##Type = mlir::tf_type::tftype##Type;


#ifdef HANDLE_TF_TYPE

//             class, enumerant, name
HANDLE_TF_TYPE(Qint8, QINT8, "qint8")
HANDLE_TF_TYPE(Qint16, QINT16, "qint16")
HANDLE_TF_TYPE(Qint32, QINT32, "qint32")
HANDLE_TF_TYPE(Quint8, QUINT8, "quint8")
HANDLE_TF_TYPE(Quint16, QUINT16, "quint16")
HANDLE_TF_TYPE(String, STRING, "string")

#ifndef HANDLE_CUSTOM_TF_TYPE
#define HANDLE_CUSTOM_TF_TYPE(class, enumerant, name) \
  HANDLE_TF_TYPE(class, enumerant, name)
#endif
HANDLE_CUSTOM_TF_TYPE(Resource, RESOURCE, "resource")
HANDLE_CUSTOM_TF_TYPE(Variant, VARIANT, "variant")
#undef HANDLE_CUSTOM_TF_TYPE

// All ref types are listed below this line and FloatRef is the first ref type.
// This helps in easily differentiating ref and non-ref types, and converting
// a type to/from ref types.

#ifndef HANDLE_TF_REF_TYPE
#define HANDLE_TF_REF_TYPE(class, enumerant, name) \
  HANDLE_TF_TYPE(class, enumerant, name)
#endif
HANDLE_TF_REF_TYPE(FloatRef, FLOAT_REF, "f32ref")
HANDLE_TF_REF_TYPE(DoubleRef, DOUBLE_REF, "f64ref")
HANDLE_TF_REF_TYPE(Uint4Ref, UINT4_REF, "uint4ref")
HANDLE_TF_REF_TYPE(Int4Ref, INT4_REF, "int4ref")
HANDLE_TF_REF_TYPE(Uint8Ref, UINT8_REF, "uint8ref")
HANDLE_TF_REF_TYPE(Int8Ref, INT8_REF, "int8ref")
HANDLE_TF_REF_TYPE(Uint16Ref, UINT16_REF, "uint16ref")
HANDLE_TF_REF_TYPE(Int16Ref, INT16_REF, "int16ref")
HANDLE_TF_REF_TYPE(Uint32Ref, UINT32_REF, "uint32ref")
HANDLE_TF_REF_TYPE(Int32Ref, INT32_REF, "int32ref")
HANDLE_TF_REF_TYPE(Uint64Ref, UINT64_REF, "uint64ref")
HANDLE_TF_REF_TYPE(Int64Ref, INT64_REF, "int64ref")
HANDLE_TF_REF_TYPE(StringRef, STRING_REF, "stringref")
HANDLE_TF_REF_TYPE(BoolRef, BOOL_REF, "boolref")
HANDLE_TF_REF_TYPE(Quint8Ref, QUINT8_REF, "quint8ref")
HANDLE_TF_REF_TYPE(Qint8Ref, QINT8_REF, "qint8ref")
HANDLE_TF_REF_TYPE(Quint16Ref, QUINT16_REF, "quint16ref")
HANDLE_TF_REF_TYPE(Qint16Ref, QINT16_REF, "qint16ref")
HANDLE_TF_REF_TYPE(Qint32Ref, QINT32_REF, "qint32ref")
HANDLE_TF_REF_TYPE(Bfloat16Ref, BFLOAT16_REF, "bfloat16ref")
HANDLE_TF_REF_TYPE(Complex64Ref, COMPLEX64_REF, "complex64ref")
HANDLE_TF_REF_TYPE(Complex128Ref, COMPLEX128_REF, "complex128ref")
HANDLE_TF_REF_TYPE(HalfRef, HALF_REF, "halfref")
HANDLE_TF_REF_TYPE(ResourceRef, RESOURCE_REF, "resourceref")
HANDLE_TF_REF_TYPE(Float8E4M3FNRef, FLOAT8_E4M3FN_REF, "float8e4m3fnref")
HANDLE_TF_REF_TYPE(Float8E5M2Ref, FLOAT8_E5M2_REF, "float8e5m2ref")

#ifndef HANDLE_LAST_TF_TYPE
#define HANDLE_LAST_TF_TYPE(class, enumerant, name) \
  HANDLE_TF_REF_TYPE(class, enumerant, name)
#endif
HANDLE_LAST_TF_TYPE(VariantRef, VARIANT_REF, "variantref")
#undef HANDLE_LAST_TF_TYPE

#undef HANDLE_TF_REF_TYPE
#undef HANDLE_TF_TYPE
#endif

}  // end namespace TF
}  // end namespace mlir

#endif

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TYPES_H_
