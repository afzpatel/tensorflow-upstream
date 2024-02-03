/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if !(defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
#define HIP_HOST_DEVICE __device__
#include "rocblas/internal/rocblas_hip_f8_impl.h"
#endif

#include <limits>

namespace stream_executor {
namespace gpu {

__global__ void rocm_castHalf2FloatKernel(float* dst, __half* src, int size) {
    for (int i = threadIdx.x + blockIdx.x * 256; i < size; i+=blockDim.x*gridDim.x)
      dst[i] = __half2float(src[i]);
}

__device__ inline __half fp8_to_half(uint32_t i32val)
{
    // upcast
#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
    float fval;
    asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return __float2half(fval);
#else
    half hval = rocblas_hip_f8_impl::cast_from_f8<3,4,_Float16,true>(i32val & 0xFF);
    return __half(hval);
#endif
}

__device__ inline __half bf8_to_half(uint32_t i32val)
{
#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
    float    fval;
    asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return __float2half(fval);
#else
    half hval = rocblas_hip_f8_impl::cast_from_f8<2,5,_Float16,true>(i32val & 0xFF);
    return __half(hval);
#endif
}

template <int T>
__global__ void rocm_castHalf2F8Kernel(uint8_t* dst, const __half* src, uint64_t size, uint32_t seed) {
    for (uint64_t i = threadIdx.x + blockIdx.x * 256; i*4 < size; i+=blockDim.x*gridDim.x) {
       uint32_t ival = 0;
       uint32_t u32_0 = *(uint32_t*)&src[4*i];
       uint32_t u32_1 = *(uint32_t*)&src[4*i+2];
       uint32_t rng = u32_0 ^ ((u32_1 >> 11) | (u32_1 << 21));
       rng *= 0x7000149;
       rng = rng ^ (i * 0x9701241f) ^ seed;
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
       if(T==0) {
         ival = __builtin_amdgcn_cvt_sr_fp8_f32(__half2float(src[4*i+0]), rng, ival, 0);
         ival = __builtin_amdgcn_cvt_sr_fp8_f32(__half2float(src[4*i+1]), (rng<<8)|(rng>>24), ival, 1);
         ival = __builtin_amdgcn_cvt_sr_fp8_f32(__half2float(src[4*i+2]), (rng<<16)|(rng>>16), ival, 2);
         ival = __builtin_amdgcn_cvt_sr_fp8_f32(__half2float(src[4*i+3]), (rng<<24)|(rng>>8), ival, 3);
       } else {
         ival = __builtin_amdgcn_cvt_sr_bf8_f32(__half2float(src[4*i+0]), rng, ival, 0);
         ival = __builtin_amdgcn_cvt_sr_bf8_f32(__half2float(src[4*i+1]), (rng<<8)|(rng>>24), ival, 1);
         ival = __builtin_amdgcn_cvt_sr_bf8_f32(__half2float(src[4*i+2]), (rng<<16)|(rng>>16), ival, 2);
         ival = __builtin_amdgcn_cvt_sr_bf8_f32(__half2float(src[4*i+3]), (rng<<24)|(rng>>8), ival, 3);
       }
       /*
       if(T==0)
          ival = __builtin_amdgcn_cvt_pk_fp8_f32(__half2float(src[2*i+0]), __half2float(src[2*i+1]), 0, false);
        else
          ival = __builtin_amdgcn_cvt_pk_bf8_f32(__half2float(src[2*i+0]), __half2float(src[2*i+1]), 0, false);
        */
#else
       const _Float16* psrc = (const _Float16*) src;
       ival = rocblas_hip_f8_impl::cast_to_f8<T?2:3,T?5:4,_Float16,true,true>(psrc[4*i+0], true, rng);
       ival |= rocblas_hip_f8_impl::cast_to_f8<T?2:3,T?5:4,_Float16,true,true>(psrc[4*i+1], true, (rng<<8)|(rng>>24)) << 8;
       ival |= rocblas_hip_f8_impl::cast_to_f8<T?2:3,T?5:4,_Float16,true,true>(psrc[4*i+2], true, (rng<<16)|(rng>>16)) << 16;
       ival |= rocblas_hip_f8_impl::cast_to_f8<T?2:3,T?5:4,_Float16,true,true>(psrc[4*i+3], true, (rng<<24)|(rng>>8)) << 24;
#endif
       *(uint32_t*)(&dst[4*i]) = ival;
    }
}

template <int T>
__global__ void rocm_castF82HalfKernel(__half* dst, const uint8_t* src, uint64_t size) {
    for (uint64_t i = threadIdx.x + blockIdx.x * 256; i*2 < size; i+=blockDim.x*gridDim.x) {
      uint16_t x = *(const uint16_t*)(&src[2*i+0]);
      __half f1, f2;
       if(T==0) {
        f1 = fp8_to_half(x);
        f2 = fp8_to_half(x>>8);
      } else {
        f1 = bf8_to_half(x);
        f2 = bf8_to_half(x>>8);
      }
      dst[2*i+0] = f1;
      dst[2*i+1] = f2;
    }
}

void rocm_castHalf2F8(void* stream, uint8_t* dst, const __half* src, uint64_t size, int fp8) {
  int x_blocks = (size + 1023) / 1024;
  if(x_blocks > 65536)
    x_blocks = 65536;
  hipLaunchKernelGGL(fp8 ? rocm_castHalf2F8Kernel<0>
                         : rocm_castHalf2F8Kernel<1>,
                     dim3(x_blocks, 1, 1),
                     256, 0, (hipStream_t)stream, dst, src, size, rand() ^ (rand() << 17));
}

void rocm_castF82Half(void* stream, __half* dst, const uint8_t* src, uint64_t size, int fp8) {
  int x_blocks = (size + 1023) / 1024;
  if(x_blocks > 65536)
    x_blocks = 65536;
  hipLaunchKernelGGL(fp8 ? rocm_castF82HalfKernel<0>
                         : rocm_castF82HalfKernel<1>,
                     dim3(x_blocks, 1, 1),
                     256, 0, (hipStream_t)stream, dst, src, size);
}


void rocm_castHaf2Float(void* stream, float* dst, __half* src, int size) {
  int x_blocks = (size + 255) / 256;
  if(x_blocks > 65536)
    x_blocks = 65536;
  hipLaunchKernelGGL(rocm_castHalf2FloatKernel,
                     dim3(x_blocks, 1, 1),
                     256, 0, (hipStream_t)stream, dst, src, size);
}

__global__ void rocm_Broadcast_fp32Kernel(float* dst, int dst_stride,
                                          int batches, float* src, int size) {
  dst += blockIdx.y * 4 * dst_stride + blockIdx.z * dst_stride * batches;
  src += blockIdx.z * size;
  float* dst2 = dst + dst_stride;
  float* dst3 = dst + dst_stride * 2;
  float* dst4 = dst + dst_stride * 3;
  bool b2 = (blockIdx.y * 4 + 1 < batches);
  bool b3 = (blockIdx.y * 4 + 2 < batches);
  bool b4 = (blockIdx.y * 4 + 3 < batches);
  for (int i = threadIdx.x + blockIdx.x * 256; i < size;
       i += blockDim.x * gridDim.x) {
    dst[i] = src[i];
    if (b2) {
      dst2[i] = src[i];
    }
    if (b3) {
      dst3[i] = src[i];
    }
    if (b4) {
      dst4[i] = src[i];
    }
  }
}

void rocm_Broadcast_fp32(void* stream, float* dst, int dst_stride, int batches,
                         int src_batches, float* src, int size) {
  int x_blocks = (size + 255) / 256;
  hipLaunchKernelGGL(rocm_Broadcast_fp32Kernel,
                     dim3(x_blocks, (batches + 3) / 4, src_batches),
                     min(256, (int)size), 0, (hipStream_t)stream, dst,
                     dst_stride, batches, src, size);
}

// GPU kernel to populate an array of pointers:
//
//   [base + stride * i for i in range(n)].
//
__global__ void __xla_MakeBatchPointers(char* base, int stride, int n,
                                        void** ptrs_out) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  ptrs_out[idx] = base + idx * stride;
}

void rocm_MakeBatchPointers(void* stream, char* base, int stride, int n,
                            void** ptrs_out) {
  const int threads_per_block = 256;
  hipLaunchKernelGGL(
      __xla_MakeBatchPointers,
      dim3((n + threads_per_block - 1) / threads_per_block, 1, 1),
      dim3(threads_per_block, 1, 1), 0, (hipStream_t)stream, base, stride, n,
      ptrs_out);
}

__device__ float sigmoid(float x) {
  if (x > 0)
    return 1. / (1. + __expf(-x));
  else
    return __expf(x) / (__expf(x) + 1.);
}

template <typename T, int act_mode>
__global__ void launchInplaceBiasActivation_kernel(T* c_data,
                                                   const T* bias_data,
                                                   uint64_t m, uint64_t n,
                                                   int64_t ldc, float param) {
  uint64_t x = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= n || y >= m) return;
  float v = static_cast<float>(c_data[x + y * ldc]) +
            static_cast<float>(bias_data[x]);
  if (act_mode == 1)
    v = sigmoid(v);
  else if (act_mode == 2)
    v = v > 0.0f ? v : 0.0f;
  else if (act_mode == 3)
    v = v > 0.0f ? (v > 6.0f ? 6.0f : v) : 0.0f;
  else if (act_mode == 4)
    v = v > 0.0f ? (v > param ? param : v) : 0.0f;
  else if (act_mode == 5)
    v = tanh(v);
  else if (act_mode == 6)
    v = v > -param ? (v > param ? param : v) : -param;
  else if (act_mode == 7)
    v = v > 0.0f ? v : __expf(v) - 1;
  else if (act_mode == 8)
    v = v > 0.0f ? v : param * v;
  else if (act_mode == 9)
    v = 0.5 * v * (1 + erf(v / sqrt(2.0f)));
  c_data[x + y * ldc] = (T)v;
}

template <typename T>
void launchInplaceBiasActivation(hipStream_t stream, void* c_data,
                                 const void* bias_data, int activation_mode,
                                 uint64_t m, uint64_t n, int64_t ldc,
                                 float param) {
  uint64_t bx = min(n, static_cast<uint64_t>(256));
  uint64_t by = min(m, static_cast<uint64_t>(256) / bx);
  uint64_t gx = (n + bx - 1) / bx;
  uint64_t gy = (m + by - 1) / by;
  auto kernel = launchInplaceBiasActivation_kernel<T, 0>;
  if (activation_mode == 1)
    kernel = launchInplaceBiasActivation_kernel<T, 1>;
  else if (activation_mode == 2)
    kernel = launchInplaceBiasActivation_kernel<T, 2>;
  else if (activation_mode == 3)
    kernel = launchInplaceBiasActivation_kernel<T, 3>;
  else if (activation_mode == 4)
    kernel = launchInplaceBiasActivation_kernel<T, 4>;
  else if (activation_mode == 5)
    kernel = launchInplaceBiasActivation_kernel<T, 5>;
  else if (activation_mode == 6)
    kernel = launchInplaceBiasActivation_kernel<T, 6>;
  else if (activation_mode == 7)
    kernel = launchInplaceBiasActivation_kernel<T, 7>;
  else if (activation_mode == 8)
    kernel = launchInplaceBiasActivation_kernel<T, 8>;
  else if (activation_mode == 9)
    kernel = launchInplaceBiasActivation_kernel<T, 9>;

  hipLaunchKernelGGL(kernel, dim3(gx, gy, 1), dim3(bx, by, 1), 0, stream,
                     static_cast<T*>(c_data), static_cast<const T*>(bias_data),
                     m, n, ldc, param);
}

template void launchInplaceBiasActivation<__half>(
    hipStream_t stream, void* c_data, const void* bias_data,
    int activation_mode, uint64_t m, uint64_t n, int64_t ldc, float param);

template void launchInplaceBiasActivation<hip_bfloat16>(
    hipStream_t stream, void* c_data, const void* bias_data,
    int activation_mode, uint64_t m, uint64_t n, int64_t ldc, float param);

template void launchInplaceBiasActivation<float>(
    hipStream_t stream, void* c_data, const void* bias_data,
    int activation_mode, uint64_t m, uint64_t n, int64_t ldc, float param);

template void launchInplaceBiasActivation<double>(
    hipStream_t stream, void* c_data, const void* bias_data,
    int activation_mode, uint64_t m, uint64_t n, int64_t ldc, float param);

};  // namespace gpu
};  // namespace stream_executor
