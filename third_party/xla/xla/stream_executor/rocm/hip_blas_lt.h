/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

#ifndef XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_
#define XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_

#include "rocm/rocm_config.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/host_or_device_scalar.h"
#include "xla/types.h"
#include "tsl/platform/status.h"

#include "rocm/rocm_config.h"
#if TF_HIPBLASLT

#include "xla/stream_executor/rocm/hip_blas_utils.h"

namespace stream_executor {

namespace gpu {
class GpuExecutor;
}  // namespace gpu

namespace rocm {

class BlasLt : public gpu::BlasLt {
  template <typename T>
  using Owned =
      std::unique_ptr<std::remove_pointer_t<T>, hipblasStatus_t (*)(T)>;

 public:
  struct MatrixLayout {
    static tsl::StatusOr<MatrixLayout> Create(const gpu::MatrixLayout& m, 
      int dynamic_range=-1, bool f8_on = false);

    hipDataType type() const { return datatype_; }
    hipblasLtMatrixLayout_t get() const { return handle_.get(); }
    uint64_t rows() const { return m_.num_rows; }
    uint64_t cols() const { return m_.num_cols; }
    uint64_t batch() const { return m_.batch_size; }

   private:
    MatrixLayout(const gpu::MatrixLayout& m, 
                 hipblasLtMatrixLayout_t handle, hipDataType datatype)
        : m_(m), handle_(handle, wrap::hipblasLtMatrixLayoutDestroy),
          datatype_(datatype) {}

    gpu::MatrixLayout m_;
    Owned<hipblasLtMatrixLayout_t> handle_;
    hipDataType datatype_;
  };

  class MatmulDesc {
   public:
    static tsl::StatusOr<MatmulDesc> Create(
        blas::ComputationType compute_type, blas::DataType scale_type,
        blas::Transpose trans_a = blas::Transpose::kNoTranspose,
        blas::Transpose trans_b = blas::Transpose::kNoTranspose,
        Epilogue epilogue = Epilogue::kDefault,
        PointerMode pointer_mode = PointerMode::kHost);

    hipblasComputeType_t compute_type() const { return compute_type_; }
    hipDataType scale_type() const { return datatype_; }
    hipblasPointerMode_t pointer_mode() const {
      return HIPBLAS_POINTER_MODE_HOST;
    }
    hipblasLtMatmulDesc_t get() const { return handle_.get(); }
    blas::Transpose get_trans_a() const { return trans_a_; }
    blas::Transpose get_trans_b() const { return trans_b_; }
    hipblasLtEpilogue_t epi() const { return epi_; }
   private:
    MatmulDesc(hipblasLtMatmulDesc_t handle, hipblasComputeType_t compute_type,
               hipDataType datatype,
               blas::Transpose trans_a, blas::Transpose trans_b)
        : handle_(handle, wrap::hipblasLtMatmulDescDestroy),
          compute_type_(compute_type),
          datatype_(datatype), trans_a_(trans_a), trans_b_(trans_b) {}

    Owned<hipblasLtMatmulDesc_t> handle_;
    hipblasComputeType_t compute_type_;
    hipDataType datatype_;
    blas::Transpose trans_a_, trans_b_;
    hipblasLtEpilogue_t epi_;
  };

  struct MatmulPlan : public gpu::BlasLt::MatmulPlan {
    MatmulPlan(const BlasLt& blas_lt_ref, MatmulDesc&& op_desc,
               MatrixLayout&& a_desc, MatrixLayout&& b_desc,
               MatrixLayout&& c_desc, MatrixLayout&& d_desc,
               xla::complex128 alpha, double beta, bool must_swap_operands,
               const int* range, 
               bool f8, bool f8_emu, bool f8_sr, bool f8_dynamic_scale, int f8_emu_param)
        : blas_lt_ref_(blas_lt_ref),
          op_desc_(std::move(op_desc)),
          a_desc_(std::move(a_desc)),
          b_desc_(std::move(b_desc)),
          c_desc_(std::move(c_desc)),
          d_desc_(std::move(d_desc)),
          alpha_(alpha),
          beta_(beta),
          must_swap_operands_(must_swap_operands),
          range1_(range[0]), range2_(range[1]), f8_(f8),
          f8_emu_(f8_emu), f8_sr_(f8_sr), f8_dynamic_scale_(f8_dynamic_scale), f8_emu_param_(f8_emu_param) {}

    ~MatmulPlan() override = default;

    tsl::Status ExecuteOnStream(
        Stream* stream, DeviceMemoryBase a_buffer, DeviceMemoryBase b_buffer,
        DeviceMemoryBase c_buffer, DeviceMemoryBase d_buffer,
        DeviceMemoryBase bias_buffer,  // may be null
        DeviceMemoryBase aux_buffer,   // may be null
        DeviceMemoryBase a_scale_buffer, DeviceMemoryBase b_scale_buffer,
        DeviceMemoryBase c_scale_buffer, DeviceMemoryBase d_scale_buffer,
        DeviceMemoryBase d_amax_buffer, const MatmulAlgorithm& algorithm,
        ScratchAllocator& scratch_allocator,
        blas::ProfileResult* profile_result = nullptr) const override;

    tsl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithms(
        size_t max_algorithm_count, size_t max_workspace_size) const override;

   protected:
    tsl::Status ValidateInputs(blas::DataType scale_type, bool alpha_on_device,
                               bool beta_on_device, blas::DataType A_type,
                               blas::DataType B_type, blas::DataType C_type,
                               blas::DataType D_type) const override;

    tsl::Status DoMatmul(Stream* stream, const void* alpha, DeviceMemoryBase a,
                         DeviceMemoryBase b, const void* beta,
                         DeviceMemoryBase c, DeviceMemoryBase d,
                         const MatmulAlgorithm& algorithm,
                         ScratchAllocator& scratch_allocator,
                         DeviceMemoryBase bias, DeviceMemoryBase aux,
                         DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
                         DeviceMemoryBase c_scale, DeviceMemoryBase d_scale,
                         DeviceMemoryBase d_amax,
                         blas::ProfileResult* profile_result) const override;

  public:
    const BlasLt& blas_lt_ref_;
    // TODO(cjfj): Add consistency checks for types, shapes, etc.?
    MatmulDesc op_desc_;
    MatrixLayout a_desc_;
    MatrixLayout b_desc_;
    MatrixLayout c_desc_;
    MatrixLayout d_desc_;
    xla::complex128 alpha_;
    double beta_;
    bool must_swap_operands_;
    int range1_, range2_;
    bool f8_, f8_emu_, f8_sr_, f8_dynamic_scale_;
    int f8_emu_param_;
    mutable std::map<std::pair<size_t, size_t>, std::vector<MatmulAlgorithm> > algos_;
    mutable std::vector<MatmulAlgorithm> last_algos_;
  };  // class MatmulPlan

  explicit BlasLt(gpu::GpuExecutor* parent)
      : parent_(parent), blas_lt_(nullptr, wrap::hipblasLtDestroy) {}

  tsl::Status Init() override;

  tsl::StatusOr<MatmulPlanPtr> GetMatmulPlan(const gpu::GemmConfig& cfg,
                                             Epilogue epilogue) const override;

  ~BlasLt() override = default;

protected:
  uint8_t* f8_staging_buffer_ = nullptr;
  uint64_t f8_staging_buffer_size_ = 512000000;

  absl::Mutex f8_mu_;
  hipEvent_t f8_staging_event_;
private:
  gpu::GpuExecutor* parent_;
  mutable absl::Mutex mu_;
  Owned<hipblasLtHandle_t> blas_lt_ ABSL_GUARDED_BY(mu_);
};

}  // namespace rocm
}  // namespace stream_executor

#endif  // TF_HIPBLASLT
#endif  // XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_
