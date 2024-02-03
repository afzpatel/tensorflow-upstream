/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/stream_executor/rocm/rocm_blas.h"

#include "xla/stream_executor/rocm/rocblas_wrapper.h"

#define EIGEN_USE_GPU
#include <assert.h>

#include <complex>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "rocm/rocm_config.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_activation.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_helpers.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_timer.h"
#include "xla/stream_executor/platform/dso_loader.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/logging.h"
#include "tsl/util/determinism.h"
#include "tsl/util/env_var.h"
using tsl::OpDeterminismRequired;

namespace stream_executor {
namespace gpu {

//PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kRocBlasPlugin);

void rocm_castHalf2F8(void* stream, uint8_t* dst, const __half* src, uint64_t size, int fp8);
void rocm_castF82Half(void* stream, __half* dst, const uint8_t* src, uint64_t size, int fp8);

std::map<std::string, uint64_t> flop_lookup;
std::map<std::string, std::vector<double> > timing_database;
int timing_database_call_count = 0;
double total_time = 0, last_total_time = 0;
std::mutex timing_mutex;

template <typename T>
float stdev(const std::vector<T>& v)
{
  float mean = 0, s = 0;
  for(int i=0; i<v.size(); i++)
    mean += float(v[i]);
  mean /= v.size();
  for(int i=0; i<v.size(); i++)
    s += (float(v[i])-mean)*(float(v[i])-mean);
  return sqrt(s/v.size());
}

template <typename T>
T filtered_mean(std::vector<T> v, int& excluded)
{
  if(v.size()==1)
  {
    excluded=0;
    return v[0];
  }
  std::sort(v.begin(), v.end());
  T mean = 0, stdev = 0;
  for(int i=(v.size()>2 ? 1 : 0); i<int(v.size()>2 ? v.size()-1 : v.size()); i++)
    mean += v[i];
  mean /= T(v.size() - (v.size()>2 ? 2 : 0));
  T mean2 = 0;
  if(v.size() > 3) 
  {
    for(int i=(v.size()>2 ? 1 : 0); i<int(v.size()>2 ? v.size()-1 : v.size()); i++)
      stdev += (v[i]-mean)*(v[i]-mean);
    stdev = sqrt(stdev/T(v.size() - (v.size()>2 ? 2 : 0)));

    int mean_count = 0;
    for(auto x: v)
      if(x>mean-5*stdev && x<mean+5*stdev)
      {
        mean2+=x;
        mean_count++;
      }
    mean2 /= mean_count;
    excluded = v.size() - mean_count;
  }
  else
  {
    mean2 = mean;
    excluded = 0;
  }
  return mean2;
}

void update_db(std::string key, uint64_t flops, double ms)
{
  int64_t do_stats = 0;
  tsl::Status status = tsl::ReadInt64FromEnvVar("GEMM_STATS", 0, &do_stats);
  if(do_stats == 0)
    return;

  timing_mutex.lock();
  total_time += ms;
  if(timing_database.find(key) == timing_database.end())
  {
    timing_database[key] = std::vector<double>{ms};
    flop_lookup[key] = flops;
  }
  else
    timing_database[key].push_back(ms);
  timing_database_call_count++;
  if(total_time > last_total_time+10000.0 || !(timing_database_call_count % 10000))
  {
    fflush(stdout);
    //double total_time = 0;
    for(auto k: timing_database)
    {
      double mean;
      int excluded;
      mean = filtered_mean(k.second, excluded);
      //total_time += mean * k.second.size();
      printf("%s   %.3f ms/call, %d calls (%d excluded), %.1f TFlops\n", 
        k.first.c_str(), mean, k.second.size(), excluded, double(flop_lookup[k.first])/(mean*1e9));
    }
    printf("%.3f s total recorded time\n", total_time/1e3);
    last_total_time = total_time;
    fflush(stdout);
  }
  timing_mutex.unlock(); 
}

extern void rocm_Broadcast_fp32(void *stream, float *dst, int dst_stride,
                                int batches, int src_batches, float *src,
                                int size);

template <class T>
const typename RocBlasTypeConversionHelper<T>::mapped_type * const* complex_cast(
    const DeviceMemory<T*> &a) {
  return reinterpret_cast<
      const typename RocBlasTypeConversionHelper<T>::mapped_type * const*>(
      GpuMemory(a));
}

template <class T>
typename RocBlasTypeConversionHelper<T>::mapped_type * const* complex_cast(
    DeviceMemory<T*> &a) {
  return reinterpret_cast<
      typename RocBlasTypeConversionHelper<T>::mapped_type * const*>(
      GpuMemory(a));
}

template <class T>
const typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    const DeviceMemory<T> &a) {
  return reinterpret_cast<
      const typename RocBlasTypeConversionHelper<T>::mapped_type *>(
      GpuMemory(a));
}

template <class T>
const typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    const T &a) {
  return reinterpret_cast<
      const typename RocBlasTypeConversionHelper<T>::mapped_type *>(&a);
}
template <class T>
typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    DeviceMemory<T> *a) {
  return reinterpret_cast<
      typename RocBlasTypeConversionHelper<T>::mapped_type *>(
      GpuMemoryMutable(a));
}

static void blas_log(const char *c) {}

static string ToString(rocblas_status status) {
  switch (status) {
    case rocblas_status_success:
      return "rocblas_status_success";
    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";
    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";
    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";
    default:
      return absl::StrCat("<invalid rocBLAS status: ", status, ">");
  }
}

bool ROCMBlas::Init() {
  gpu::ScopedActivateExecutorContext sac{parent_};
  rocblas_status ret = wrap::rocblas_create_handle(&blas_);
  if (ret != rocblas_status_success) {
    LOG(ERROR) << "failed to create rocBLAS handle: " << ToString(ret);
    return false;
  }

#if TF_HIPBLASLT
  if (!blas_lt_.Init().ok()) {
    LOG(ERROR) << "Failed to initialize hipblasLt";
    return false;
  }
#endif

  int dev = 0;
  hipError_t result = hipGetDevice(&dev);
  hipDeviceProp_t props;
  result = hipGetDeviceProperties(&props, dev);
  if (result == hipSuccess) {
    std::string gcnArchName = props.gcnArchName;
    auto pos = gcnArchName.find(":");
    if (pos != string::npos) gcnArchName = gcnArchName.substr(0, pos);
    pos = gcnArchName.find("gfx");
    if (pos != string::npos) gcnArchName = gcnArchName.substr(pos + 3);
    if ((gcnArchName == "908") || 
        (gcnArchName == "90a") || (gcnArchName == "940") ||
        (gcnArchName == "941") || (gcnArchName == "942"))
      has_mfma_ = true;
    if ((gcnArchName == "940") ||
        (gcnArchName == "941") || (gcnArchName == "942"))
      has_f8_ = true;
    if(gcnArchName == "90a")
      use_hgemm_alt_impl_ = true;
  }

  return true;
}

ROCMBlas::ROCMBlas(gpu::GpuExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)),
      blas_(nullptr)
#if TF_HIPBLASLT
      ,
      blas_lt_(parent)
#endif
{
}

ROCMBlas::~ROCMBlas() {
  if (blas_ != nullptr) {
    gpu::ScopedActivateExecutorContext sac{parent_};
    wrap::rocblas_destroy_handle(blas_);
  }
}

bool ROCMBlas::SetStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsGpuStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  gpu::ScopedActivateExecutorContext sac{parent_};

  rocblas_status ret =
      wrap::rocblas_set_stream(blas_, AsGpuStreamValue(stream));
  if (ret != rocblas_status_success) {
    LOG(ERROR) << "failed to set stream for rocBLAS calls: " << ToString(ret);
    return false;
  }

  return true;
}

hipStream_t ROCMBlas::ROCMStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsGpuStreamValue(stream) != nullptr);
  gpu::ScopedActivateExecutorContext sac{parent_};
  return AsGpuStreamValue(stream);
}

namespace {

// Helper functions transforming blas arguments into rocBLAS arguments.

rocblas_operation ROCMBlasTranspose(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return rocblas_operation_none;
    case blas::Transpose::kTranspose:
      return rocblas_operation_transpose;
    case blas::Transpose::kConjugateTranspose:
      return rocblas_operation_conjugate_transpose;
    default:
      LOG(FATAL) << "Invalid value of blas::Transpose.";
  }
}

rocblas_fill ROCMBlasUpperLower(blas::UpperLower uplo) {
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return rocblas_fill_upper;
    case blas::UpperLower::kLower:
      return rocblas_fill_lower;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

rocblas_diagonal ROCMBlasDiagonal(blas::Diagonal diag) {
  switch (diag) {
    case blas::Diagonal::kUnit:
      return rocblas_diagonal_unit;
    case blas::Diagonal::kNonUnit:
      return rocblas_diagonal_non_unit;
    default:
      LOG(FATAL) << "Invalid value of blas::Diagonal.";
  }
}

rocblas_side ROCMBlasSide(blas::Side side) {
  switch (side) {
    case blas::Side::kLeft:
      return rocblas_side_left;
    case blas::Side::kRight:
      return rocblas_side_right;
    default:
      LOG(FATAL) << "Invalid value of blas::Side.";
  }
}

}  // namespace

template <typename FuncT, typename... Args>
bool ROCMBlas::DoBlasInternalImpl(FuncT rocblas_func, Stream *stream,
                                  bool pointer_mode_host, bool err_on_failure,
                                  Args... args) {
  absl::MutexLock lock{&mu_};

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return false;
  }

  gpu::ScopedActivateExecutorContext sac{parent_};

  // set the atomics mode, leaving default to library
  bool allow_atomics = !OpDeterminismRequired();
  rocblas_status ret;
  if (!allow_atomics) {
    ret = wrap::rocblas_set_atomics_mode(blas_, rocblas_atomics_not_allowed);
    if (err_on_failure && ret != rocblas_status_success) {
      LOG(ERROR) << "failed to to set atomics mode before "
                 << rocblas_func.kName << ": " << ToString(ret);
    }
  }

  ret = rocblas_func(blas_, args...);
  if (err_on_failure && ret != rocblas_status_success) {
    LOG(ERROR) << "failed to run ROCBLAS routine " << rocblas_func.kName << ": "
               << ToString(ret);
  }
  return ret == rocblas_status_success;
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  blas_log("DoBlasAxpy");
  return DoBlasInternal(wrap::rocblas_saxpy, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  blas_log("DoBlasAxpy");
  return DoBlasInternal(wrap::rocblas_daxpy, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_caxpy, stream, /* pointer_mode_host = */ true, elem_count,
      complex_cast(alpha), complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_zaxpy, stream, /* pointer_mode_host = */ true, elem_count,
      complex_cast(alpha), complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_scopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_dcopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_ccopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_zcopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  blas_log("DoBlasScal<float>");
  return DoBlasInternal(wrap::rocblas_sscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_dscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_csscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_zdscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_cscal, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(alpha), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_zscal, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(alpha), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_sgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_dgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_cgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  blas_log("DoBlasGemv\n");
  return DoBlasInternal(
      wrap::rocblas_zgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_ssbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, k, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_dsbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, k, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

static void maybe_start_timer(std::optional<GpuTimer>& timer, Stream *stream)
{
  int64_t do_stats = 0;
  tsl::Status status = tsl::ReadInt64FromEnvVar("GEMM_STATS", 0, &do_stats);
  if(do_stats) {
    hipStreamCaptureStatus captureStatus;
    hipError_t err = hipStreamIsCapturing(AsGpuStreamValue(stream), &captureStatus);
    if(err == hipSuccess && captureStatus == hipStreamCaptureStatusNone) {
      auto timer_or_status = gpu::GpuTimer::Create(AsGpuStream(stream));
      if (!timer_or_status.ok()) {
        LOG(ERROR) << "Failed to create timer";
        return;
      }
      timer.emplace(std::move(*timer_or_status));
    }
  }
}

static void maybe_stop_timer(std::optional<GpuTimer>& timer, 
              std::string report_string, uint64_t flops)
{
    // At time intervals we're working with (often <200 microseconds), synchronization
    // can have observable negative impact on overall timing.
    if(timer) {
      tsl::StatusOr<absl::Duration> elapsed = timer->GetElapsedDuration();
      if(elapsed.ok()) {
        double t = absl::ToDoubleMilliseconds(*elapsed);
        update_db(report_string, flops, t);
      }
    }
}

int fetch(std::vector<Eigen::half>& v, const Eigen::half* p, int n, hipStream_t stream)
{
  v.resize(n);
  hipMemcpyAsync(&v[0], p, n*2, hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);
  int ninf=0;
  for(int i=0; i<n; i++)
    if(!isfinite(float(v[i])))
      ninf++;
  return ninf;
}

void eval_differences(const std::vector<Eigen::half>& va, const std::vector<Eigen::half>& vb, double& mean, double& maxval)
{
  double sum=0;
  maxval=0;
  int count=0;
  for(int i=0; i<va.size(); i++)
  {
    if(!isfinite(float(va[i])) || !isfinite(float(vb[i])))
      continue;
    count++;
    double d = fabs(float(va[i])-float(vb[i]));
    sum+=d;
    maxval=std::max(maxval,d);
  }
  if(count==0)
    count=1;
  mean = sum/count;
}

static void read_f8_env_flags(const NumericOptions& numeric_options, bool& f8_on, bool& f8_emu, bool has_f8_)
{
  f8_on = !(numeric_options.grad_flags & 4);
  int64_t f8_env = 0, f8_mm_env = 0, f8_emu_int = 0;
  tsl::Status status;
  status = tsl::ReadInt64FromEnvVar("TF_ROCM_F8", 0, &f8_env);
  status = tsl::ReadInt64FromEnvVar("F8_MM", 0, &f8_mm_env);
  status = tsl::ReadInt64FromEnvVar("F8_EMU", 0, &f8_emu_int);
  f8_emu = (f8_emu_int != 0);
  if(f8_env == 0 || (f8_mm_env & 1) == 0)
    f8_on = false;
  if(f8_on && !has_f8_)
    f8_emu = true;
}

/**
 * 
 *  ALPHA/BETA TYPES
 * 
 * For half and bf16, alpha and beta point to floats.
 * For all other types, alpha and beta point to values of the same type as a/b/c.
 * 
 * On the rocblas side, non-ex functions expect the same type as a/b/c 
 *    (this seems to be a deviation from the blas standard);
 *    and ex functions expect the same type as the compute type (i.e. floats.)
 *    We wrap all ex calls into objects that do the casting.
 * 
**/

tsl::Status ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                                 blas::Transpose transb, uint64_t m, uint64 n,
                                 uint64_t k, blas::DataType dtype,
                                 const void *alpha, const DeviceMemoryBase &a,
                                 int lda, const DeviceMemoryBase &b, int ldb,
                                 const void *beta, DeviceMemoryBase *c, int ldc,
                                 const NumericOptions &numeric_options) {
  blas_log("DoBlasGemm");
  VLOG(1) << absl::StreamFormat(
      "doing rocBLAS GEMM: at=%d bt=%d m=%u n=%u "
      "k=%llu alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);
  if (dtype == blas::DataType::kHalf || dtype == blas::DataType::kFloat) {
    if (transa == blas::Transpose::kNoTranspose) {
      if (lda < static_cast<int64_t>(m)) {
        LOG(WARNING) << "GEMM lda was smaller than m (no transpose case); "
                        "precondition violation";
      }
    } else {
      if (lda < static_cast<int64_t>(k)) {
        LOG(WARNING) << "GEMM lda (" << lda << ") was smaller than k (" << k
                     << ") (transpose case); precondition violation";
      }
    }
    if (transb == blas::Transpose::kNoTranspose) {
      if (ldb < static_cast<int64_t>(k)) {
        LOG(WARNING) << "GEMM ldb (" << ldb << ") was smaller than k (" << k
                     << ") (no transpose case); precondition violation";
      }
    } else {
      if (ldb < static_cast<int64_t>(n)) {
        LOG(WARNING) << "GEMM ldb was smaller than n (transpose case); "
                        "precondition violation";
      }
    }
  }
  if(!(numeric_options.grad_flags & NumericOptions::GF_Initialized)) {
    printf("ERROR: DoBlasGemm with uninitialized gradient flags\n");
    exit(-1);
  }

  tsl::Status status;
  std::optional<gpu::GpuTimer> timer;
  std::string prefix;
  if(dtype==blas::DataType::kHalf)
    prefix="F16";
  else if(dtype==blas::DataType::kBF16)
    prefix="BF16";
  else if(dtype==blas::DataType::kFloat)
    prefix="F32";
  std::string report_string = " matmul M " + std::to_string(m)
    + " N " + std::to_string(n) + " K " + std::to_string(k) + " "
    + ((transa == blas::Transpose::kNoTranspose) ? "N" : "T")
    + ((transb == blas::Transpose::kNoTranspose) ? "N" : "T");

  uint32_t gemm_ex_flags = rocblas_gemm_flags_none;
  bool is_backprop = (numeric_options.grad_flags & 3);
  if (is_backprop && use_hgemm_alt_impl_)
    gemm_ex_flags = rocblas_gemm_flags_fp16_alt_impl;

  int64_t gemm_numerics = 0;
  status = tsl::ReadInt64FromEnvVar("GEMM_NUMERICS", 0, &gemm_numerics);

  Eigen::half alpha_half, beta_half;

  const void * alpha_downcast = alpha, *beta_downcast = beta;
  if(dtype == blas::DataType::kHalf) {
    alpha_half = Eigen::half(*static_cast<const float *>(alpha));
    beta_half = Eigen::half(*static_cast<const float *>(beta));
    alpha_downcast = &alpha_half;
    beta_downcast = &beta_half;
  }

  /* I would like to specify the type with a template parameter, but that
   * is a C++20 extension and can't be enabled (the compiler does support it,
   * but it messes with Eigen. */
  auto call_gemm = [&](auto func, auto unity) {
         return DoBlasInternalStatus(
            func, stream, /* pointer_mode_host = */ true,
            ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
            reinterpret_cast<const decltype(unity)*>(alpha_downcast),
            reinterpret_cast<const decltype(unity)*>(a.opaque()), lda,
            reinterpret_cast<const decltype(unity)*>(b.opaque()), ldb,
            reinterpret_cast<const decltype(unity)*>(beta_downcast),
            reinterpret_cast<decltype(unity)*>(c->opaque()), ldc);
  };

  auto call_gemm_ex = [&](rocblas_datatype dt, void* pa=0, void* pb=0, void* pc=0, void* pd=0) {
      return DoBlasInternalStatus(
          wrap::rocblas_gemm_ex, stream, /* pointer_mode_host = */ true,
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb),
          (rocblas_int)m, (rocblas_int)n, (rocblas_int)k,
          alpha, 
          pa?pa:a.opaque(), dt, lda, 
          pb?pb:b.opaque(), dt, ldb, beta, 
          pc?pc:c->opaque(), dt, ldc, 
          pd?pd:c->opaque(), dt, ldc, 
          rocblas_datatype_f32_r,
          rocblas_gemm_algo_standard, 0, gemm_ex_flags);
  };

#if ROCBLAS_VERSION_MAJOR>3 || (ROCBLAS_VERSION_MAJOR==3 && ROCBLAS_VERSION_MINOR>=1)
  if(dtype==blas::DataType::kHalf) {
    if(!(numeric_options.grad_flags & 256) ) {
      printf("DoBlasGemm: no F8 flags!\n");
      exit(0);
    }

    std::unique_ptr<TemporaryDeviceMemory<uint8_t> > temp_mem;
    DeviceMemory<uint8_t> device_memory;
    bool f8_on, f8_emu;
    read_f8_env_flags(numeric_options, f8_on, f8_emu, has_f8_);
    rocblas_computetype compute_type;

    if(f8_on) {
      report_string += " f8_flags " + std::to_string(numeric_options.grad_flags);
      if(f8_emu)
        report_string += " emulated ";

      switch (numeric_options.grad_flags & 3) {
        case 0:
          compute_type = rocblas_compute_type_f8_f8_f32;
          break;
        case 1:
          compute_type = rocblas_compute_type_bf8_f8_f32;
          break;
        case 2:
          compute_type = rocblas_compute_type_f8_bf8_f32;
          break;
        case 3:
          return tsl::errors::Internal(absl::StrCat("Unexpected grad_flags for GEMM: ",
                                            numeric_options.grad_flags & 3));
      }
      report_string = "F8" + report_string + " " + std::to_string(numeric_options.grad_flags & 3);

      uint8_t *temp_a, *temp_b;

      TF_ASSIGN_OR_RETURN(temp_mem, 
        stream->AllocateTemporaryArray<uint8_t>(m*k+n*k));
      device_memory = DeviceMemory<uint8_t>(*(temp_mem->mutable_device_memory()));

      temp_a = (uint8_t*)device_memory.opaque();
      temp_b = temp_a+m*k;
      bool a_fp8 = (numeric_options.grad_flags & 3) != 1;
      bool b_fp8 = (numeric_options.grad_flags & 3) != 2;

      rocm_castHalf2F8(AsGpuStreamValue(stream), temp_a, (const __half*)a.opaque(), m*k, a_fp8 ? 1 : 0);
      rocm_castHalf2F8(AsGpuStreamValue(stream), temp_b, (const __half*)b.opaque(), n*k, b_fp8 ? 1 : 0);
      //hipStreamSynchronize(AsGpuStreamValue(stream));

      maybe_start_timer(timer, stream);
      tsl::Status retval;
      if(f8_emu) {
        rocm_castF82Half(AsGpuStreamValue(stream), (__half*)a.opaque(), temp_a, m*k, a_fp8 ? 1 : 0);
        rocm_castF82Half(AsGpuStreamValue(stream), (__half*)b.opaque(), temp_b, n*k, b_fp8 ? 1 : 0);
        retval = call_gemm_ex(rocblas_datatype_f16_r);
      }
      else {
        retval = DoBlasInternalStatus(
            wrap::rocblas_gemm_ex3, stream, /*ignored*/ true,
            ROCMBlasTranspose(transa), ROCMBlasTranspose(transb),
            (rocblas_int)m, (rocblas_int)n, (rocblas_int)k,
            alpha, temp_a, a_fp8 ? rocblas_datatype_f8_r : rocblas_datatype_bf8_r, lda,
            temp_b, b_fp8 ? rocblas_datatype_f8_r : rocblas_datatype_bf8_r, ldb,
            beta, 
            c->opaque(), rocblas_datatype_f16_r, ldc,
            c->opaque(), rocblas_datatype_f16_r, ldc,
            compute_type,
            rocblas_gemm_algo_standard, 0, 0);
      }
  #if 0
        if(gemm_numerics)
        {
          hipStreamSynchronize(AsGpuStreamValue(stream));
          int ainf = fetch(va, (const Eigen::half*)a.opaque(), m*k, AsGpuStreamValue(stream));
          int binf = fetch(vb, (const Eigen::half*)b.opaque(), n*k, AsGpuStreamValue(stream));
          int cinf8 = fetch(vc8, (const Eigen::half*)c->opaque(), m*n, AsGpuStreamValue(stream));

          hipError = hipStreamSynchronize(AsGpuStreamValue(stream));

          int dev;
          hipGetDevice(&dev);
          printf("<%d> %d  %e %e -> %e (f8) / %e (f16)\n", dev, numeric_options.grad_flags & 3, stdev(va), stdev(vb), stdev(vc8), stdev(vc));

          if(ainf || binf || cinf8)
          {
            printf("<%d> infinities in gemm_ex %s (%d %d %d); inputs %e %e\n", dev, report_string.c_str(), ainf, binf, cinf8, stdev(va), stdev(vb));
            //printf("ERROR: non-finite output from rocblas_gemm_ex3 (%s)\n", report_string.c_str());
            /*
            */
            //if(!isfinite(float(check[2][0]))) 
            if((numeric_options.grad_flags & 3) == 0)
            {
              printf("Inputs:\nA\tB\n");
              for(int i=0; i<4; i++)
                printf("%e %e\n", float(va[i]), float(vb[i]));
              printf("Outputs:\nF8\tF16\n");
              for(int i=0; i<4; i++)
                printf("%04x %04x  %e %e\n", *(uint16_t*)(&vc8[i]), *(uint16_t*)(&vc[i]), 
                  float(vc8[i]), float(vc[i]));
              double mean, maxval;
              eval_differences(vc, vc8, mean, maxval);
              printf("Mean difference: %e, max difference: %e\n", mean, maxval);
              fflush(stdout);

              printf("Nonfinite: %d (A) %d (B) %d (C-f8)\n", ainf, binf, cinf8);

              float* pa32, *pb32, *pc32;
              hipMalloc((void**)&pa32, va.size()*4);
              hipMalloc((void**)&pb32, vb.size()*4);
              hipMalloc((void**)&pc32, vc.size()*4);
              for(int i=0; i<va.size(); i++)
                pa32[i] = float(va[i]);
              for(int i=0; i<vb.size(); i++)
                pb32[i] = float(vb[i]);
              float alpha32=1.0f, beta32=0.0f;

              status = call_gemm_ex(rocblas_datatype_f32_r, pa32, pb32, pc32, pc32);
              hipError = hipStreamSynchronize(AsGpuStreamValue(stream));

              for(int i=0; i<vc.size(); i++)
              {
                  if(!isfinite(float(vc[i])) || !isfinite(float(vc8[i])))
                  {
                    printf("%d   %04x %04x  %e %e  %e\n", i, *(uint16_t*)&vc[i], *(uint16_t*)&vc8[i], 
                       float(vc[i]), float(vc8[i]), pc32[i]);
                    break;
                  }
              }          
              fflush(stdout);
              exit(0);
            }
          }
        }
  #endif
      maybe_stop_timer(timer, report_string, m*n*k*2);
      return retval;
    }
  }
#endif // ROCBLAS_VERSION_MAJOR>3 || (ROCBLAS_VERSION_MAJOR==3 && ROCBLAS_VERSION_MINOR>=1) 
  report_string = prefix + report_string;

  maybe_start_timer(timer, stream);
  switch (dtype) {
    case blas::DataType::kHalf:
      if (has_mfma_) {
        status = call_gemm_ex(rocblas_datatype_f16_r);
      } else {
        status = call_gemm(wrap::rocblas_hgemm, rocblas_half());
      }
      break;
    case blas::DataType::kBF16:
      status = call_gemm_ex(rocblas_datatype_bf16_r);
      break;
    case blas::DataType::kFloat:
      status = call_gemm(wrap::rocblas_sgemm, 1.0f);
      break;
    case blas::DataType::kDouble:
      status = call_gemm(wrap::rocblas_dgemm, 1.0);
      break;
    case blas::DataType::kComplexFloat: {
      status = call_gemm(wrap::rocblas_cgemm, rocblas_float_complex());
      break;
    }
    case blas::DataType::kComplexDouble: {
      status = call_gemm(wrap::rocblas_zgemm, rocblas_double_complex());
      break;
    }
    default:
      status = tsl::errors::Internal("Unsupported datatype for GEMM: ",
                                   blas::DataTypeString(dtype));
  }
  maybe_stop_timer(timer, report_string, m*n*k*2);
  return status;
}

tsl::Status ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, const void *beta, DeviceMemoryBase *c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const NumericOptions &numeric_options,
    blas::ProfileResult *output_profile_result) {
  // ROCM TODO: properly implement the interface
  return tsl::errors::Internal("DoBlasGemmWithAlgorithm ",
                               "is not implemented on ROCm yet");
}

tsl::Status ROCMBlas::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, int64_t stride_a, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, blas::DataType type_c, int ldc, int64_t stride_c,
    int batch_count, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const NumericOptions &numeric_options,
    blas::ProfileResult *output_profile_result) {
  // ROCM TODO: properly implement the interface
  return tsl::errors::Internal("DoBlasGemmStridedBatchedWithAlgorithm ",
                               "is not implemented on ROCm yet");
}

bool ROCMBlas::GetBlasGemmAlgorithms(
    Stream *stream, std::vector<blas::AlgorithmType> *out_algorithms) {
  // ROCM TODO: properly implement the interface
  return true;
}

struct MemoryCopyOp {
  char *src_ptr;
  char *dst_ptr;
  uint64_t size;
  uint64_t count;
  uint64_t dst_stride;
  uint64_t src_count;
};

// Check whether two Memory Copy Ops can be fold together.
// If it's true, fold it. Otherwise, return false.
static bool MemCopyOpsFold(MemoryCopyOp &y, const MemoryCopyOp &x) {
  bool misaligned = (x.size & 3) ||
                    (reinterpret_cast<uint64_t>(x.dst_ptr) & 3) ||
                    (reinterpret_cast<uint64_t>(x.src_ptr) & 3) ||
                    (reinterpret_cast<uint64_t>(y.dst_ptr) & 3) ||
                    (reinterpret_cast<uint64_t>(y.src_ptr) & 3);

  int64_t dst_step = reinterpret_cast<int64_t>(x.dst_ptr) -
                     reinterpret_cast<int64_t>(y.dst_ptr);

  if (x.src_ptr == y.src_ptr && x.size == y.size &&
      (y.count == 1 || x.dst_ptr == y.dst_ptr + y.count * y.dst_stride) &&
      !misaligned && y.src_count == 1 && !(dst_step & 3)) {
    if (y.count == 1) {
      y.dst_stride = dst_step;
    }
    y.count++;
    return true;
  } else if (x.src_ptr == y.src_ptr + y.size &&
             x.dst_ptr == y.dst_ptr + y.size && y.count == 1 &&
             y.src_count == 1) {
    y.size += x.size;
    return true;
  }
  if (x.src_ptr == y.src_ptr + y.size * y.src_count &&
      x.dst_ptr == y.dst_ptr + y.dst_stride * y.src_count * y.count &&
      x.count == y.count && x.dst_stride == y.dst_stride) {
    y.src_count += x.src_count;
    return true;
  }
  return false;
}

// This copies from source memory: raw_ptrs[i] to target memory:
// device_memory_ptr at the interval of matrix_byte_size, or vice versa.
// The below algorithm tries to minimize the number of memcpy by consolidating
// neighboring memcpy into a single request.
template <typename MAPPED_T>
tsl::Status ReorganizeMemory(Stream *stream,
                             DeviceMemory<MAPPED_T> *device_memory,
                             const std::vector<MAPPED_T *> &raw_ptrs,
                             int batch_count, uint64_t batch_stride,
                             bool gather) {
  if (gather == false) {
    return tsl::Status(absl::StatusCode::kUnimplemented,
                       "gather=false is unsupported");
  }

  assert(batch_count > 0);
  char *device_memory_ptr = static_cast<char *>(device_memory->opaque());
  char *src_ptr = reinterpret_cast<char *>(raw_ptrs[0]);
  char *dst_ptr = device_memory_ptr;
  size_t matrix_byte_size = batch_stride * sizeof(MAPPED_T);

  std::vector<MemoryCopyOp> mem_copy_ops{
      MemoryCopyOp{src_ptr, dst_ptr, matrix_byte_size, 1, 0, 1}};

  for (int i = 1; i < batch_count; ++i) {
    src_ptr = reinterpret_cast<char *>(raw_ptrs[i]);
    dst_ptr = device_memory_ptr + i * matrix_byte_size;

    MemoryCopyOp x{src_ptr, dst_ptr, matrix_byte_size, 1, 0, 1};
    while (mem_copy_ops.size() > 1 &&
           MemCopyOpsFold(mem_copy_ops[mem_copy_ops.size() - 2],
                          mem_copy_ops.back())) {
      mem_copy_ops.pop_back();
    }
    MemoryCopyOp &op = mem_copy_ops.back();
    if (MemCopyOpsFold(op, x)) {
      continue;
    }
    mem_copy_ops.push_back(x);
  }

  while (mem_copy_ops.size() > 1 &&
         MemCopyOpsFold(mem_copy_ops[mem_copy_ops.size() - 2],
                        mem_copy_ops.back())) {
    mem_copy_ops.pop_back();
  }

  int i = 0;
  for (auto &x : mem_copy_ops) {
    if (x.src_count > 1 || x.count > 1) {
      rocm_Broadcast_fp32(AsGpuStreamValue(stream),
                          reinterpret_cast<float *>(x.dst_ptr),
                          x.dst_stride >> 2, x.count, x.src_count,
                          reinterpret_cast<float *>(x.src_ptr), x.size >> 2);
    } else {
      DeviceMemoryBase src_mem = DeviceMemoryBase(x.src_ptr, x.size);
      DeviceMemoryBase target_mem = DeviceMemoryBase(x.dst_ptr, x.size);
      bool a_status = stream->ThenMemcpy(&target_mem, src_mem, x.size).ok();
      if (!a_status) {
        return tsl::Status(
            absl::StatusCode::kInternal,
            "failed to copy device memory in ROCMBlas::DoBlasGemmBatched");
      }
    }
    i++;
  }
  return tsl::OkStatus();
}

template <typename T>
tsl::Status ROCMBlas::AllocateStridedBuffer(
    const std::vector<typename RocBlasTypeConversionHelper<T>::mapped_type *>
        &raw_ptrs,
    int batch_count, uint64_t batch_stride, ScratchAllocator *scratch_allocator,
    Stream *stream,
    std::unique_ptr<TemporaryDeviceMemory<
        typename RocBlasTypeConversionHelper<T>::mapped_type>> *temp_memory,
    DeviceMemory<typename RocBlasTypeConversionHelper<T>::mapped_type>
        *device_memory,
    bool copy_data, bool &reallocated) {
  assert(device_memory != nullptr);

  using MAPPED_T = typename RocBlasTypeConversionHelper<T>::mapped_type;

  bool needs_allocate_strided = false;
  for (int i = 1; i < batch_count; ++i) {
    uint64_t tmp_batch_stride = raw_ptrs[i] - raw_ptrs[i - 1];
    if (tmp_batch_stride != batch_stride) {
      needs_allocate_strided = true;
      break;
    }
  }

  size_t matrix_byte_size = batch_stride * sizeof(MAPPED_T);
  size_t matrix_batch_byte_size = matrix_byte_size * batch_count;

  // No need to do re-allocation, take the short cut and return
  if (!needs_allocate_strided) {
    *device_memory = DeviceMemory<MAPPED_T>(
        DeviceMemoryBase(raw_ptrs[0], matrix_batch_byte_size));
    reallocated = false;
    return tsl::OkStatus();
  }

  if (scratch_allocator != nullptr) {
    TF_ASSIGN_OR_RETURN(
        DeviceMemory<uint8> batch_matrix_bytes,
        scratch_allocator->AllocateBytes(matrix_batch_byte_size));
    *device_memory = DeviceMemory<MAPPED_T>(batch_matrix_bytes);
  } else {
    assert(temp_memory != nullptr);
    TF_ASSIGN_OR_RETURN(*temp_memory, stream->AllocateTemporaryArray<MAPPED_T>(
                                          matrix_batch_byte_size));
    *device_memory =
        DeviceMemory<MAPPED_T>(*(*temp_memory)->mutable_device_memory());
  }
  assert(batch_count > 0);
  char* device_memory_ptr = static_cast<char*>(device_memory->opaque());
  char* src_ptr = reinterpret_cast<char*>(raw_ptrs[0]);
  char* dst_ptr = device_memory_ptr;
  uint64_t cur_stride_size = matrix_byte_size;

  reallocated = true;

  if (copy_data)
    return ReorganizeMemory(stream, device_memory, raw_ptrs, batch_count,
                            batch_stride, true);
  return tsl::OkStatus();
}

template <typename T, typename FuncT>
tsl::Status ROCMBlas::DoBlasGemmBatchedInternal(
    FuncT rocblas_func, Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64_t m, uint64 n, uint64 k, T alpha,
    DeviceMemorySlice<T> a_ptrs_to_wrappers, int lda,
    DeviceMemorySlice<T> b_ptrs_to_wrappers, int ldb, T beta,
    DeviceMemorySlice<T> c_ptrs_to_wrappers, int ldc, int batch_count,
    const NumericOptions& numeric_options,
    ScratchAllocator *scratch_allocator) {
  using MAPPED_T = typename RocBlasTypeConversionHelper<T>::mapped_type;

  // Sanity checks before making any further progress
  uint64_t batch_stride_a = 0;
  uint64_t batch_stride_b = 0;
  uint64_t batch_stride_c = 0;

  assert(ldc >= m);
  batch_stride_c = ldc * n;

  if (ROCMBlasTranspose(transa) == rocblas_operation_none) {
    assert(lda >= m);
    batch_stride_a = lda * k;
  } else {
    assert(lda >= k);
    batch_stride_a = lda * m;
  }

  if (ROCMBlasTranspose(transb) == rocblas_operation_none) {
    assert(ldb >= k);
    batch_stride_b = ldb * n;
  } else {
    assert(ldb >= n);
    batch_stride_b = ldb * k;
  }

  // Allocate local vectors to hold device pointers to matrices
  std::vector<MAPPED_T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  for (int i = 0; i < batch_count; ++i) {
    // static_cast does work when converting Eigen::half* to rocblas_half*,
    // hence the use of reinterpret_cast
    a_raw_ptrs.push_back(
        reinterpret_cast<MAPPED_T *>(a_ptrs_to_wrappers[i]->opaque()));
    b_raw_ptrs.push_back(
        reinterpret_cast<MAPPED_T *>(b_ptrs_to_wrappers[i]->opaque()));
    c_raw_ptrs.push_back(
        reinterpret_cast<MAPPED_T *>(c_ptrs_to_wrappers[i]->opaque()));
  }

  DeviceMemory<MAPPED_T> a;
  // Make sure the temporary memory are in-scope before the function returns
  std::unique_ptr<TemporaryDeviceMemory<MAPPED_T>> a_temp;
  bool reallocated_a, reallocated_b, reallocated_c;
  tsl::Status a_allocation_status = AllocateStridedBuffer<T>(
      a_raw_ptrs, batch_count, batch_stride_a, scratch_allocator, stream,
      &a_temp, &a, true, reallocated_a);
  if (a_allocation_status != tsl::OkStatus()) {
    return a_allocation_status;
  }

  DeviceMemory<MAPPED_T> b;
  std::unique_ptr<TemporaryDeviceMemory<MAPPED_T>> b_temp;
  tsl::Status b_allocation_status = AllocateStridedBuffer<T>(
      b_raw_ptrs, batch_count, batch_stride_b, scratch_allocator, stream,
      &b_temp, &b, true, reallocated_b);
  if (b_allocation_status != tsl::OkStatus()) {
    return b_allocation_status;
  }

  DeviceMemory<MAPPED_T> c;
  std::unique_ptr<TemporaryDeviceMemory<MAPPED_T>> c_temp;
  tsl::Status c_allocation_status = AllocateStridedBuffer<T>(
      c_raw_ptrs, batch_count, batch_stride_c, scratch_allocator, stream,
      &c_temp, &c, true, reallocated_c);  // can disable copy if beta=0
  if (c_allocation_status != tsl::OkStatus()) {
    return c_allocation_status;
  }

  std::string prefix;
  if (std::is_same_v<T, Eigen::half>)
    prefix="F16";
  else if (std::is_same_v<T, Eigen::bfloat16>)
    prefix="BF16";
  else if (std::is_same_v<T, float>)
    prefix="F32";
  std::string report_string = " matmul B " + std::to_string(batch_count)
    + " M " + std::to_string(m)
    + " N " + std::to_string(n) + " K " + std::to_string(k) 
    + " "
    + ((transa == blas::Transpose::kNoTranspose) ? "N" : "T")
    + ((transb == blas::Transpose::kNoTranspose) ? "N" : "T")
    + " " + rocblas_func.kName;

  std::optional<gpu::GpuTimer> timer;
  maybe_start_timer(timer, stream);
  tsl::Status retval;

  bool ok;
  MAPPED_T *alpha_ptr = reinterpret_cast<MAPPED_T *>(&alpha);
  MAPPED_T *beta_ptr = reinterpret_cast<MAPPED_T *>(&beta);
  ok = DoBlasInternal(rocblas_func, stream, /* pointer_mode_host = */ true,
                      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m,
                      n, k, GpuComplex(alpha_ptr), GpuMemory(a), lda,
                      batch_stride_a, GpuMemory(b), ldb, batch_stride_b,
                      GpuComplex(beta_ptr), GpuMemoryMutable(&c), ldc,
                      batch_stride_c, batch_count);
  maybe_stop_timer(timer, report_string, uint64_t(batch_count)*m*n*k*2);
  if (!ok)
    return tsl::Status(absl::StatusCode::kInternal,
                       "failed BLAS call, see log for details");

  if (reallocated_c)
    return ReorganizeMemory(stream, &c, c_raw_ptrs, batch_count, batch_stride_c,
                            false);
  return tsl::OkStatus();
}

class rocblas_hgemm_strided_batched_mfma {
  int ALT_;
public:
  rocblas_hgemm_strided_batched_mfma(int ALT) : ALT_(ALT) {}
  std::string kName = "rocblas_hgemm_strided_batched_mfma";
  rocblas_status operator()(rocblas_handle      handle,
                                                  rocblas_operation   transA,
                                                  rocblas_operation   transB,
                                                  rocblas_int         m,
                                                  rocblas_int         n,
                                                  rocblas_int         k,
                                                  const rocblas_half* alpha,
                                                  const rocblas_half* A,
                                                  rocblas_int         lda,
                                                  rocblas_stride      stride_a,
                                                  const rocblas_half* B,
                                                  rocblas_int         ldb,
                                                  rocblas_stride      stride_b,
                                                  const rocblas_half* beta,
                                                  rocblas_half*       C,
                                                  rocblas_int         ldc,
                                                  rocblas_stride      stride_c,
                                                  rocblas_int         batch_count) {
  float alpha32 = float(*(const __half*)alpha);
  float beta32 = float(*(const __half*)beta);
  uint32_t flags = rocblas_gemm_flags_none;
#if TF_ROCM_VERSION >= 50000  
  if(ALT_)
    flags = rocblas_gemm_flags_fp16_alt_impl;
#endif
  return wrap::rocblas_gemm_strided_batched_ex(handle,
      transA, transB,
      m, n, k,
      &alpha32, 
      A, rocblas_datatype_f16_r, lda, stride_a,
      B, rocblas_datatype_f16_r, ldb, stride_b,
      &beta32,
      C, rocblas_datatype_f16_r, ldc, stride_c,
      C, rocblas_datatype_f16_r, ldc, stride_c,
      batch_count,
      rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard,
      0,
      flags);
}
};


class rocblas_hgemm_strided_batched_ex3 {
  int grad_flags_;
  Stream* stream_;
  bool emulated_;
  bool ALT_;
public:
  rocblas_hgemm_strided_batched_ex3(int grad_flags, Stream* stream, bool emu, bool alt) : grad_flags_(grad_flags), stream_(stream), emulated_(emu), ALT_(alt) {}
  std::string kName = "rocblas_hgemm_strided_batched_ex3";
  rocblas_status operator()(rocblas_handle      handle,
                                                  rocblas_operation   transA,
                                                  rocblas_operation   transB,
                                                  rocblas_int         m,
                                                  rocblas_int         n,
                                                  rocblas_int         k,
                                                  const rocblas_half* alpha,
                                                  const rocblas_half* A,
                                                  rocblas_int         lda,
                                                  rocblas_stride      batch_stride_a,
                                                  const rocblas_half* B,
                                                  rocblas_int         ldb,
                                                  rocblas_stride      batch_stride_b,
                                                  const rocblas_half* beta,
                                                  rocblas_half*       C,
                                                  rocblas_int         ldc,
                                                  rocblas_stride      batch_stride_c,
                                                  rocblas_int         batch_count) {
  float alpha32 = float(*(const __half*)alpha);
  float beta32 = float(*(const __half*)beta);
  uint32_t flags = rocblas_gemm_flags_none;

  rocblas_computetype compute_type;
  switch (grad_flags_ & 3) {
    case 0:
      compute_type = rocblas_compute_type_f8_f8_f32;
      break;
    case 1:
      compute_type = rocblas_compute_type_bf8_f8_f32;
      break;
    case 2:
      compute_type = rocblas_compute_type_f8_bf8_f32;
      break;
    case 3:
      printf("Unexpected grad_flags for GEMM: %d\n", grad_flags_ & 3);
      return rocblas_status_invalid_value;
  }

  if(batch_stride_a!=m*k || batch_stride_b!=n*k) {
    printf("Warning: unexpected buffer dimensions (Strided Batched %c%c): m=%d n=%d k=%d batch_stride_a=%d batch_stride_b=%d batch_count=%d\n",
       transA==rocblas_operation_none ? 'N' : 'T',
       transB==rocblas_operation_none ? 'N' : 'T',
       m, n, k, batch_stride_a, batch_stride_b, batch_count);
  }

  std::unique_ptr<TemporaryDeviceMemory<uint8_t> > temp_mem;
  DeviceMemory<uint8_t> device_memory;
  auto status = stream_->AllocateTemporaryArray<uint8_t>((batch_stride_a+batch_stride_b)*batch_count);
  if(status.ok())
    temp_mem = std::move(status).value();
  else
    return rocblas_status_memory_error;
  device_memory = DeviceMemory<uint8_t>(*(temp_mem->mutable_device_memory()));

  bool a_fp8 = (grad_flags_ & 3) != 1;
  bool b_fp8 = (grad_flags_ & 3) != 2;
  uint8_t* temp_a = (uint8_t*)device_memory.opaque();
  uint8_t* temp_b = temp_a+batch_stride_a*batch_count;

  rocm_castHalf2F8(AsGpuStreamValue(stream_), temp_a, (const __half*)A, batch_stride_a*batch_count, a_fp8 ? 1 : 0);
  rocm_castHalf2F8(AsGpuStreamValue(stream_), temp_b, (const __half*)B, batch_stride_b*batch_count, b_fp8 ? 1 : 0);

  if(emulated_) {
      rocm_castF82Half(AsGpuStreamValue(stream_), (__half*)A, temp_a, batch_stride_a*batch_count, a_fp8 ? 1 : 0);
      rocm_castF82Half(AsGpuStreamValue(stream_), (__half*)B, temp_b, batch_stride_b*batch_count, b_fp8 ? 1 : 0);
      uint32_t flags = rocblas_gemm_flags_none;
      if(ALT_)
        flags = rocblas_gemm_flags_fp16_alt_impl;
      return wrap::rocblas_gemm_strided_batched_ex(handle,
        transA, transB,
        m, n, k,
        &alpha32, 
        A, rocblas_datatype_f16_r, lda, batch_stride_a,
        B, rocblas_datatype_f16_r, ldb, batch_stride_b,
        &beta32,
        C, rocblas_datatype_f16_r, ldc, batch_stride_c,
        C, rocblas_datatype_f16_r, ldc, batch_stride_c,
        batch_count,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard,
        0,
        flags);
  }

  return wrap::rocblas_gemm_strided_batched_ex3(handle,
      transA, transB,
      m, n, k,
      &alpha32,
      temp_a, a_fp8 ? rocblas_datatype_f8_r : rocblas_datatype_bf8_r, lda, batch_stride_a,
      temp_b, b_fp8 ? rocblas_datatype_f8_r : rocblas_datatype_bf8_r, ldb, batch_stride_b,
      &beta32,
      C, rocblas_datatype_f16_r, ldc, batch_stride_c,
      C, rocblas_datatype_f16_r, ldc, batch_stride_c,
      batch_count,
      compute_type,
      rocblas_gemm_algo_standard,
      0,
      rocblas_gemm_flags_none);
}
};

class rocblas_gemm_strided_batched_bf16 {
public:
  std::string kName = "rocblas_gemm_strided_batched_bf16";
  rocblas_status operator()(
      rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
      rocblas_int m, rocblas_int n, rocblas_int k,
      const rocblas_bfloat16 *alpha, const rocblas_bfloat16 *A, rocblas_int lda,
      rocblas_stride stride_a, const rocblas_bfloat16 *B, rocblas_int ldb,
      rocblas_stride stride_b, const rocblas_bfloat16 *beta, rocblas_bfloat16 *C,
      rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count) {
    float alpha32 = float(*(const Eigen::bfloat16 *)alpha);
    float beta32 = float(*(const Eigen::bfloat16 *)beta);
    uint32_t flags = rocblas_gemm_flags_none;
    //printf("rocblas_gemm_strided_batched_ex(bf16) %f %f\n", alpha32, beta32);
    return wrap::rocblas_gemm_strided_batched_ex(
        handle, transA, transB, m, n, k, &alpha32, A, rocblas_datatype_bf16_r,
        lda, stride_a, B, rocblas_datatype_bf16_r, ldb, stride_b, &beta32, C,
        rocblas_datatype_bf16_r, ldc, stride_c, C, rocblas_datatype_bf16_r, ldc,
        stride_c, batch_count, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, flags);
  }
};

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha, DeviceMemorySlice<Eigen::half> a,
    int lda, DeviceMemorySlice<Eigen::half> b, int ldb, float beta,
    DeviceMemorySlice<Eigen::half> c, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator) {
  blas_log("DoBlasGemmBatched");
  const Eigen::half alpha_half(alpha);
  const Eigen::half beta_half(beta);
  tsl::Status status;
  //auto func = wrap::rocblas_hgemm_strided_batched;
  if(!(numeric_options.grad_flags & NumericOptions::GF_Initialized)) {
    printf("ERROR: DoBlasGemmBatched with uninitialized gradient flags\n");
    exit(-1);
  }

  bool f8_on, f8_emu;
  read_f8_env_flags(numeric_options, f8_on, f8_emu, has_f8_);
  auto call_gemm = [&](auto x) { return DoBlasGemmBatchedInternal(
        x,
        stream, transa, transb, m, n, k,
        alpha_half, a, lda, b, ldb, beta_half, c, ldc, batch_count,
        numeric_options,
        scratch_allocator);
  };

  bool is_backprop = (numeric_options.grad_flags & 3);
  if (f8_on) {
    status = call_gemm(rocblas_hgemm_strided_batched_ex3(numeric_options.grad_flags, stream, f8_emu, is_backprop && use_hgemm_alt_impl_));
  } else if (has_mfma_) {
    status = call_gemm(rocblas_hgemm_strided_batched_mfma(is_backprop && use_hgemm_alt_impl_));
  } else {
    status = call_gemm(wrap::rocblas_hgemm_strided_batched);
  }

  if (!status.ok()) {
    LOG(ERROR) << status;
  }

  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha,
    DeviceMemorySlice<Eigen::bfloat16> a_array, int lda,
    DeviceMemorySlice<Eigen::bfloat16> b_array, int ldb, float beta,
    DeviceMemorySlice<Eigen::bfloat16> c_array, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator) {
  blas_log("DoBlasGemmBatched");
  printf("DoBlasGemmBatched<bf16> %f %f\n", alpha, beta);
  const Eigen::bfloat16 alpha_bf16(alpha);
  const Eigen::bfloat16 beta_bf16(beta);

  tsl::Status status = DoBlasGemmBatchedInternal(
      rocblas_gemm_strided_batched_bf16(), stream, transa, transb, m, n, k,
      alpha_bf16, a_array, lda, b_array, ldb, beta_bf16, c_array, ldc,
      batch_count, numeric_options, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

#define IMPL_DoBlasGemmBatched(T, Fun) \
bool ROCMBlas::DoBlasGemmBatched(                                                 \
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,   \
    uint64_t n, uint64 k, T alpha, DeviceMemorySlice<T> a_array,                  \
    int lda, DeviceMemorySlice<T> b_array, int ldb, T beta,                       \
    DeviceMemorySlice<T> c_array, int ldc, int batch_count,                       \
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator) { \
  tsl::Status status = DoBlasGemmBatchedInternal(                                 \
      Fun, stream, transa, transb, m, n, k,                                       \
      alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,         \
      numeric_options, scratch_allocator);                                        \
  if (!status.ok()) {                                                             \
    LOG(ERROR) << status;                                                         \
  }                                                                               \
  return status.ok();                                                             \
}

IMPL_DoBlasGemmBatched(float, wrap::rocblas_sgemm_strided_batched)
IMPL_DoBlasGemmBatched(double, wrap::rocblas_dgemm_strided_batched)
IMPL_DoBlasGemmBatched(std::complex<float>, wrap::rocblas_cgemm_strided_batched)
IMPL_DoBlasGemmBatched(std::complex<double>, wrap::rocblas_zgemm_strided_batched)

#define IMPL_DoBlasTrsm(T, Fun, Fun2) \
bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,                        \
                          blas::UpperLower uplo, blas::Transpose transa,          \
                          blas::Diagonal diag, uint64_t m, uint64 n,              \
                          T alpha, const DeviceMemory<T> &a, int lda,             \
                          DeviceMemory<T> *b, int ldb) {                          \
  return DoBlasInternal(Fun, stream,                                              \
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),       \
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),      \
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(a), lda,  \
                        GpuMemoryMutable(b), ldb);                                \
}                                                                                 \
                                                                                  \
bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,                 \
                                 blas::UpperLower uplo, blas::Transpose transa,   \
                                 blas::Diagonal diag, uint64_t m, uint64 n,       \
                                 T alpha, const DeviceMemory<T *> &as,            \
                                 int lda, DeviceMemory<T *> *bs, int ldb,         \
                                 int batch_count) {                               \
  return DoBlasInternal(Fun2, stream,                                             \
                        true /* = pointer_mode_host */, ROCMBlasSide(side),       \
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),      \
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(as),      \
                        lda, GpuMemoryMutable(bs), ldb, batch_count);             \
}                                                                                 \


IMPL_DoBlasTrsm(float, wrap::rocblas_strsm, wrap::rocblas_strsm_batched)
IMPL_DoBlasTrsm(double, wrap::rocblas_dtrsm, wrap::rocblas_dtrsm_batched)

#define IMPL_DoBlasTrsm_cpx(T, Fun, Fun2) \
bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,                        \
                          blas::UpperLower uplo, blas::Transpose transa,          \
                          blas::Diagonal diag, uint64_t m, uint64 n,              \
                          T alpha, const DeviceMemory<T> &a, int lda,             \
                          DeviceMemory<T> *b, int ldb) {                          \
  return DoBlasInternal(Fun, stream,                                              \
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),       \
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),      \
                        ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),        \
                        complex_cast(a), lda, complex_cast(b), ldb);              \
}                                                                                 \
                                                                                  \
bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,                 \
                                 blas::UpperLower uplo, blas::Transpose transa,   \
                                 blas::Diagonal diag, uint64_t m, uint64 n,       \
                                 T alpha,                       \
                                 const DeviceMemory<T *> &as,   \
                                 int lda,                                         \
                                 DeviceMemory<T *> *bs,         \
                                 int ldb, int batch_count) {                      \
  return DoBlasInternal(                                                          \
      Fun2, stream, true /* = pointer_mode_host */,        \
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),    \
      ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),                          \
      complex_cast(as), lda,              \
      complex_cast(*bs), ldb,             \
      batch_count);                       \
}

IMPL_DoBlasTrsm_cpx(std::complex<float>, wrap::rocblas_ctrsm, wrap::rocblas_ctrsm_batched)
IMPL_DoBlasTrsm_cpx(std::complex<double>, wrap::rocblas_ztrsm, wrap::rocblas_ztrsm_batched)

tsl::Status ROCMBlas::DoBlasGemmStridedBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, blas::DataType dtype, const void *alpha,
    const DeviceMemoryBase &a, int lda, int64_t stride_a,
    const DeviceMemoryBase &b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, int ldc, int64_t stride_c, int batch_count,
    const NumericOptions &numeric_options) {
  VLOG(1) << absl::StreamFormat(
      "doing rocBLAS SGEMM Strided Batched<float>: at=%d bt=%d m=%u n=%u "
      "k=%llu alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);
  if(!(numeric_options.grad_flags & NumericOptions::GF_Initialized)) {
    printf("ERROR: DoBlasGemmStridedBatched with uninitialized gradient flags\n");
    exit(-1);
  }


  tsl::Status status;
  bool f8_on, f8_emu;
  read_f8_env_flags(numeric_options, f8_on, f8_emu, has_f8_);

  std::optional<gpu::GpuTimer> timer;
  maybe_start_timer(timer, stream);
  std::string prefix;
  if (dtype==blas::DataType::kHalf) {
    if(f8_on && !f8_emu)
      prefix="F8";
    else if(f8_on && f8_emu)
      prefix="F8EMU";
    else
      prefix="F16";
  }
  else if (dtype==blas::DataType::kBF16)
    prefix="BF16";
  else if (dtype==blas::DataType::kFloat)
    prefix="F32";
  std::string report_string = prefix + " matmul BS " + std::to_string(batch_count)
    + " M " + std::to_string(m)
    + " N " + std::to_string(n) + " K " + std::to_string(k) 
    + " "
    + ((transa == blas::Transpose::kNoTranspose) ? "N" : "T")
    + ((transb == blas::Transpose::kNoTranspose) ? "N" : "T");

  Eigen::half alpha_half, beta_half;
  Eigen::bfloat16 alpha_bf16, beta_bf16;
  if(dtype == blas::DataType::kHalf) {
    alpha_half = Eigen::half(*static_cast<const float *>(alpha));
    beta_half = Eigen::half(*static_cast<const float *>(beta));
    alpha = &alpha_half;
    beta = &beta_half;
  } else if(dtype == blas::DataType::kBF16) {
    alpha_bf16 = Eigen::bfloat16(*static_cast<const float *>(alpha));
    beta_bf16 = Eigen::bfloat16(*static_cast<const float *>(beta));
    alpha = &alpha_bf16;
    beta = &beta_bf16;
  }

  auto call_gemm = [&](auto func, auto unity) { return DoBlasInternalStatus(
            func, stream,
            false, /* pointer_mode_host */
            ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
            reinterpret_cast<const decltype(unity)*>(alpha),
            reinterpret_cast<const decltype(unity)*>(a.opaque()), lda, stride_a,
            reinterpret_cast<const decltype(unity)*>(b.opaque()), ldb, stride_b,
            reinterpret_cast<const decltype(unity)*>(beta),
            reinterpret_cast<decltype(unity)*>(c->opaque()), ldc, stride_c,
            batch_count);
  };

  switch (dtype) {
    case blas::DataType::kHalf: {
      bool is_backprop = (numeric_options.grad_flags & 3);

      if (f8_on) {
        status = call_gemm(rocblas_hgemm_strided_batched_ex3(numeric_options.grad_flags, stream, f8_emu, is_backprop && use_hgemm_alt_impl_), rocblas_half());
      }
      else if (has_mfma_) {
        status = call_gemm(rocblas_hgemm_strided_batched_mfma(is_backprop && use_hgemm_alt_impl_), rocblas_half());
      } else {
        status = call_gemm(wrap::rocblas_hgemm_strided_batched, rocblas_half());
      }
      break;
    }
    case blas::DataType::kBF16:
    {
      status = call_gemm(rocblas_gemm_strided_batched_bf16(), rocblas_bfloat16());
      break;
    }
    case blas::DataType::kFloat:
    {
      status = call_gemm(wrap::rocblas_sgemm_strided_batched, 1.0f);
      break;
    }
    case blas::DataType::kDouble:
    {
      status = call_gemm(wrap::rocblas_dgemm_strided_batched, 1.0);
      break;
    }
    case blas::DataType::kComplexFloat: 
    {
      status = call_gemm(wrap::rocblas_cgemm_strided_batched, rocblas_float_complex());
      break;
    }
    case blas::DataType::kComplexDouble:
    {
      status = call_gemm(wrap::rocblas_zgemm_strided_batched, rocblas_double_complex());
      break;
    }
    default:
      return tsl::errors::Internal(absl::StrCat(
          "Unsupported datatype for GEMM: ", blas::DataTypeString(dtype)));
  }

  maybe_stop_timer(timer, report_string, uint64_t(batch_count)*m*n*k*2);
  return status;
}

tsl::Status ROCMBlas::GetVersion(string *version) {
  return tsl::errors::Unimplemented("");
}

}  // namespace gpu

void initialize_rocblas() {
  auto rocBlasAlreadyRegistered = PluginRegistry::Instance()->HasFactory(
      rocm::kROCmPlatformId, PluginKind::kBlas);

  if (!rocBlasAlreadyRegistered) {
    tsl::Status status =
        PluginRegistry::Instance()
            ->RegisterFactory<PluginRegistry::BlasFactory>(
                rocm::kROCmPlatformId, "rocBLAS",
                [](internal::StreamExecutorInterface *parent)
                    -> blas::BlasSupport * {
                  gpu::GpuExecutor *rocm_executor =
                      dynamic_cast<gpu::GpuExecutor *>(parent);
                  if (rocm_executor == nullptr) {
                    LOG(ERROR)
                        << "Attempting to initialize an instance of the "
                           "rocBLAS "
                        << "support library with a non-ROCM StreamExecutor";
                    return nullptr;
                  }

                  gpu::ROCMBlas *blas = new gpu::ROCMBlas(rocm_executor);
                  if (!blas->Init()) {
                    // Note: Init() will log a more specific error.
                    delete blas;
                    return nullptr;
                  }
                  return blas;
                });

    if (!status.ok()) {
      LOG(ERROR) << "Unable to register rocBLAS factory: " << status.message();
    }
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_rocblas,
                            { stream_executor::initialize_rocblas(); });
