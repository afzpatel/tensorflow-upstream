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

__device__ inline __half2 hmax(__half2 a, __half2 b)
{
  __half2 retval;
    asm volatile("v_pk_max_f16 %0, %1, %2" : "=v"(retval) : "v"(a), "v"(b)); 
    return retval;
}

__device__ inline __half2 hmin(__half2 a, __half2 b)
{
  __half2 retval;
    asm volatile("v_pk_min_f16 %0, %1, %2" : "=v"(retval) : "v"(a), "v"(b)); 
    return retval;
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

template <int T, int SR>
__device__ void castHalf2F8(uint8_t* dst, const __half* src, uint32_t seed, uint64_t i, float mult) {
     uint32_t ival = 0;
     uint32_t u32_0 = *(uint32_t*)&src[0];
     //uint32_t u32_1 = *(uint32_t*)&src[2];
     uint32_t rng = u32_0;
     __half2 hmult2{__half(mult),__half(mult)};
     if(SR) {
       //uint32_t rng = u32_0 ^ ((u32_1 >> 11) | (u32_1 << 21));
       rng *= 0x7000149;
       rng = rng ^ (i * 0x9701241f) ^ seed;
    }
     __half f8min = __half(-240.0f), f8max = __half(240.0f);
     __half2 f8min2(f8min,f8min), f8max2(f8max,f8max);
     for(int j=0; j<1; j++) {
       __half s0 = src[4*j+0];
       __half s1 = src[4*j+1];
       __half s2 = src[4*j+2];
       __half s3 = src[4*j+3];
        __half2 s01(s0, s1);
        __half2 s23(s2, s3);
        s01 *= hmult2;
        s23 *= hmult2;

        s01 = hmax(s01, f8min2);
        s23 = hmax(s23, f8min2);
        s01 = hmin(s01, f8max2);
        s23 = hmin(s23, f8max2);
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
       if(T==0) {
        // __builtin_amdgcn_fmed3f
        if(SR) {
          ival = __builtin_amdgcn_cvt_sr_fp8_f32(__half2float(s01.x), rng, ival, 0);
          ival = __builtin_amdgcn_cvt_sr_fp8_f32(__half2float(s01.y), (rng<<8)|(rng>>24), ival, 1);
          ival = __builtin_amdgcn_cvt_sr_fp8_f32(__half2float(s23.x), (rng<<16)|(rng>>16), ival, 2);
          ival = __builtin_amdgcn_cvt_sr_fp8_f32(__half2float(s23.y), (rng<<24)|(rng>>8), ival, 3);
        } else {
          ival = __builtin_amdgcn_cvt_pk_fp8_f32(__half2float(s01.x), __half2float(s01.y), 0, false); 
          ival = __builtin_amdgcn_cvt_pk_fp8_f32(__half2float(s23.x), __half2float(s23.y), ival, true); 
        }
      } else {
        if(SR) {
          ival = __builtin_amdgcn_cvt_sr_bf8_f32(__half2float(s01.x), rng, ival, 0);
          ival = __builtin_amdgcn_cvt_sr_bf8_f32(__half2float(s01.y), (rng<<8)|(rng>>24), ival, 1);
          ival = __builtin_amdgcn_cvt_sr_bf8_f32(__half2float(s23.x), (rng<<16)|(rng>>16), ival, 2);
          ival = __builtin_amdgcn_cvt_sr_bf8_f32(__half2float(s23.y), (rng<<24)|(rng>>8), ival, 3);
        } else {
          ival = __builtin_amdgcn_cvt_pk_bf8_f32(__half2float(s01.x), __half2float(s01.y), 0, false); 
          ival = __builtin_amdgcn_cvt_pk_bf8_f32(__half2float(s23.x), __half2float(s23.y), ival, true); 
        }
      }
#else
       ival = rocblas_hip_f8_impl::cast_to_f8<T?2:3,T?5:4,_Float16,true,true>( s01.x, true, rng);
       ival |= rocblas_hip_f8_impl::cast_to_f8<T?2:3,T?5:4,_Float16,true,true>(s01.y, true, (rng<<8)|(rng>>24)) << 8;
       ival |= rocblas_hip_f8_impl::cast_to_f8<T?2:3,T?5:4,_Float16,true,true>(s23.x, true, (rng<<16)|(rng>>16)) << 16;
       ival |= rocblas_hip_f8_impl::cast_to_f8<T?2:3,T?5:4,_Float16,true,true>(s23.y, true, (rng<<24)|(rng>>8)) << 24;
#endif
       *(uint32_t*)(&dst[4*j]) = ival;
     }
}

template <int T, int SR>
__global__ void rocm_castHalf2F8Kernel(uint8_t* dst, const __half* src, uint64_t size, uint32_t seed, float mult) {
    for (uint64_t i = threadIdx.x + blockIdx.x * 256; i*4 < size; i+=blockDim.x*gridDim.x) {
      castHalf2F8<T,SR>(dst+4*i, src+4*i, seed, i, mult);
    }
}

template <int T, int SR>
__global__ void rocm_castHalf2F8Kernel_2x(uint8_t* dst, const __half* src, uint8_t* dst2, const __half* src2, 
    uint64_t size, uint64_t size2, uint32_t seed, float mult, float mult2) {
    //size = (size+3)&~3;
    for (uint64_t i = threadIdx.x + blockIdx.x * 256; i*4 < size+size2; i+=blockDim.x*gridDim.x) {
      uint8_t* pdst;
      const __half* psrc;
      if(i*4<size) {
        pdst = dst+4*i;
        psrc = src+4*i;
        castHalf2F8<T,SR>(pdst, psrc, seed, i, mult);
      }
      else {
        pdst = dst2+(4*i-size);
        psrc = src2+(4*i-size);
        castHalf2F8<T,SR>(pdst, psrc, seed, i, mult2);
      }
    }
}

template <int T>
__global__ void rocm_castF82HalfKernel(__half* dst, const uint8_t* src, uint64_t size, float mult) {
    __half hmult = __half(mult);
    for (uint64_t i = threadIdx.x + blockIdx.x * 256; i*2 < size; i+=blockDim.x*gridDim.x) {
      uint16_t x = *(const uint16_t*)(&src[2*i+0]);
      __half f1, f2;
       if(T==0) {
        f1 = fp8_to_half(x) * hmult;
        f2 = fp8_to_half(x>>8) * hmult;
      } else {
        f1 = bf8_to_half(x) * hmult;
        f2 = bf8_to_half(x>>8) * hmult;
      }
      dst[2*i+0] = f1;
      dst[2*i+1] = f2;
    }
}

void rocm_castHalf2F8(void* stream, uint8_t* dst, const __half* src, uint64_t size, int fp8, float mult, bool sr) {
  int x_blocks = (size + 1023) / 1024;
  if(x_blocks > 65536)
    x_blocks = 65536;
  hipLaunchKernelGGL(sr ? (fp8 ? rocm_castHalf2F8Kernel<0,1>
                               : rocm_castHalf2F8Kernel<1,1>)
                        : (fp8 ? rocm_castHalf2F8Kernel<0,0>
                               : rocm_castHalf2F8Kernel<1,0>),
                     dim3(x_blocks, 1, 1),
                     256, 0, (hipStream_t)stream, dst, src, size, rand() ^ (rand() << 17), mult);
}

void rocm_castHalf2F8_2x(void* stream, uint8_t* dst, const __half* src, uint8_t* dst2, const __half* src2,
    uint64_t size, uint64_t size2, int fp8, float mult, float mult2, bool sr) {
  int x_blocks = (size + size2 + 1023) / 1024;
  if(x_blocks > 65536)
    x_blocks = 65536;
  hipLaunchKernelGGL(sr ? (fp8 ? rocm_castHalf2F8Kernel_2x<0,1>
                               : rocm_castHalf2F8Kernel_2x<1,1>)
                        : (fp8 ? rocm_castHalf2F8Kernel_2x<0,0>
                               : rocm_castHalf2F8Kernel_2x<1,0>),
                     dim3(x_blocks, 1, 1),
                     256, 0, (hipStream_t)stream, dst, src, dst2, src2, size, size2, rand() ^ (rand() << 17), mult, mult2);
}

void rocm_castF82Half(void* stream, __half* dst, const uint8_t* src, uint64_t size, int fp8, float mult) {
  int x_blocks = (size + 511) / 512;
  if(x_blocks > 65536)
    x_blocks = 65536;
  hipLaunchKernelGGL(fp8 ? rocm_castF82HalfKernel<0>
                         : rocm_castF82HalfKernel<1>,
                     dim3(x_blocks, 1, 1),
                     256, 0, (hipStream_t)stream, dst, src, size, mult);
}


void rocm_castHaf2Float(void* stream, float* dst, __half* src, int size) {
  int x_blocks = (size + 255) / 256;
  if(x_blocks > 65536)
    x_blocks = 65536;
  hipLaunchKernelGGL(rocm_castHalf2FloatKernel,
                     dim3(x_blocks, 1, 1),
                     256, 0, (hipStream_t)stream, dst, src, size);
}


__global__ void rocm_randomize_kernel(__half* dst, const __half* src, int n, int param, int seed)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < n) {
        //float fs = float(src[tid]);
        uint16_t i = *(uint16_t*)&src[tid];
        int exponent = (i >> 10) & 0x1F;
        if(exponent>=param && exponent<=param+3)
          *(uint16_t*)&dst[tid] = i;
     /*
     uint32_t rng = *(uint32_t*)&src[tid & ~3];
     rng *= 0x7000149;
     rng = rng ^ (tid * 0x9701241f) ^ seed;
    if(src[tid] == __half(0.0f))
      dst[tid] = src[tid];
    else if(param == 0) {
      uint32_t ival = __builtin_amdgcn_cvt_pk_fp8_f32(__half2float(src[tid]), 0.0f, 0, false); 
      dst[tid] = fp8_to_half(ival);
    } else {
      float fs = float(src[tid]);
      // 32: apply only saturation at 240
      // 22: don't change values >=128
      // 21: don't change values >=64
      // ...
      if(fabs(fs) > 240.0f)
        dst[tid] = __half(240.0f * (fs>0 ? 1.0f : -1.0f));
      else {
        uint16_t i = *(uint16_t*)&src[tid];
        int exponent = (i >> 10) & 0x1F;
        if(param == exponent) {
            param = 6;
            // reproduces true f8 for param=6
            uint32_t mask = 1 << param;
            uint16_t round_bit = i & mask;
            uint16_t drop_bits = i & (mask-1);
            uint16_t last_bit = i & (2*mask);
            i &= (0x10000-2*mask);
            if(round_bit && drop_bits)
              i += 2*mask-1;
            else if((i & mask) && !(i & (mask - 1)))
              i += last_bit ? (4*mask-1) : 0x00;
        }
        *(uint16_t*)&dst[tid] = i;
      }
        //*(uint16_t*)&dst[tid] = *(uint16_t*)&src[tid] & (0xFFFFu << param);
    }
      */
  }
}


void rocm_randomize(void* stream, __half* dst, const __half* src, uint64_t size, int param)
{
  int x_blocks = (size + 255) / 256;
  hipLaunchKernelGGL(rocm_randomize_kernel,
                     dim3(x_blocks, 1, 1),
                     256, 0, (hipStream_t)stream, dst, src, size, param, rand() ^ (rand() << 17));
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


__global__ void variance_gpu_kernel(const __half* p, int n, float* pOut) 
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  float sums[4]={0, 0, 0, 0};
  for (int i = tid; i < n; i += stride) {
    float t = float(p[i]);
    sums[0] += (t != 0.0f);
    sums[1] += t*t;
    sums[2] = max<float>(sums[2], fabs(t));
    sums[3] += t;
  }
  __shared__ float buffer[256];
  for(int iter=0; iter<4; iter++) {
    buffer[threadIdx.x] = sums[iter];
    __syncthreads();
    for (int i = 128; i > 0; i >>= 1) {
      if (threadIdx.x < i) {
        if(iter==2)
          buffer[threadIdx.x] = std::max<float>(buffer[threadIdx.x], buffer[threadIdx.x + i]);
        else
          buffer[threadIdx.x] += buffer[threadIdx.x + i];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      pOut[blockIdx.x+iter*gridDim.x] = buffer[0];
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void variance2_gpu_kernel(const T* p1, const T* p2, int n, float* pOut) 
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  float sums[4]={0, 0, -1e10f, 0};
  int maxidx=0;
  if(std::is_same_v<T, __half>) {
    for (int i = tid; i < n; i += stride) {
      float t = float(p1[i])-float(p2[i]);
      sums[0] += 1.0f;//(t != 0.0f);
      sums[1] += t*t;
      if(fabs(t)>sums[2]) {
        sums[2] = fabs(t);
        maxidx = i;
      }
      sums[3] += float(t);
    }
  } else {
    for (int i = tid; i < n; i += stride) {
      sums[0] += float(p1[i]);
      sums[1] += float(p1[i+n]);
      if(float(p1[i+2*n]) > sums[2]) {
        sums[2] = float(p1[i+2*n]);
        maxidx = *(const int*)&p1[i+4*n];
      }
      sums[3] += float(p1[i+3*n]);
    }
  }
  __shared__ float buffer[256];
  __shared__ int maxidx_buffer[256];
  for(int iter=0; iter<4; iter++) {
    buffer[threadIdx.x] = sums[iter];
    if(iter == 2)
      maxidx_buffer[threadIdx.x] = maxidx;
    __syncthreads();
    for (int i = 128; i > 0; i >>= 1) {
      if (threadIdx.x < i) {
        if(iter==2) {
          if(buffer[threadIdx.x + i] > buffer[threadIdx.x]) {
            buffer[threadIdx.x] = buffer[threadIdx.x + i];
            maxidx_buffer[threadIdx.x] = maxidx_buffer[threadIdx.x + i];
          }
        }
        else
          buffer[threadIdx.x] += buffer[threadIdx.x + i];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      pOut[blockIdx.x+iter*gridDim.x] = buffer[0];
      if(iter == 2)
        *(int*)(&pOut[blockIdx.x+4*gridDim.x]) = maxidx_buffer[0];
    }
    __syncthreads();
  }
}

float variance_gpu(void* stream, const __half* p, int n, float* maxval, float* mean) {
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  float* pTemp;
  hipMalloc(&pTemp, gridSize*4*sizeof(float));
  //variance_gpu_kernel<<<gridSize, blockSize>>>(p, n, pTemp);
  hipLaunchKernelGGL(variance_gpu_kernel, gridSize, blockSize, 0, (hipStream_t)stream,
    p, n, pTemp);
  float* pOut = new float[gridSize*4];
  hipMemcpyAsync(pOut, pTemp, gridSize*4*sizeof(float), hipMemcpyDeviceToHost, (hipStream_t)stream);
  hipStreamSynchronize((hipStream_t)stream);
  float sum = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
  for (int i = 0; i < gridSize; i++) {
    sum += pOut[i];
    sum2 += pOut[i + gridSize];
    sum3 = std::max<float>(sum3, pOut[i + 2*gridSize]);
    sum4 += pOut[i + 3*gridSize];
  }
  delete[] pOut;
  hipFree(pTemp);
  if(sum==0)
    sum=1;
  *maxval = sum3;
  *mean = sum4 / n;
  return sqrt(sum2 / sum); // - (sum / n) * (sum / n);
}

float variance2_gpu(const __half* p1, const __half* p2, int n, float* maxval, float* mean, int* maxidx) {
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  //int gridSize2 = (gridSize + blockSize - 1) / blockSize;
  float* pTemp, *pTemp2;
  hipMalloc(&pTemp, gridSize*5*sizeof(float));
  hipMalloc(&pTemp2, gridSize*5*sizeof(float));
  variance2_gpu_kernel<__half> <<<gridSize, blockSize>>>(p1, p2, n, pTemp); // produces 'gridSize' entries
  while(gridSize > 1) {
//    gridSize = gridSize2;
    int gridSize2 = (gridSize + blockSize - 1) / blockSize;
    variance2_gpu_kernel<float> <<<gridSize2, blockSize>>>(pTemp, 0, gridSize, pTemp2);
    gridSize = gridSize2;
    float* pTemp3 = pTemp;
    pTemp = pTemp2;
    pTemp2 = pTemp3;
  }
  float out[5];
  hipMemcpy(out, pTemp, 5*sizeof(float), hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  *maxval = out[2];
  *maxidx = *(int*)&out[4];
  *mean = out[3] / n;

  hipFree(pTemp);
  hipFree(pTemp2);
  return sqrt(out[1] / out[0]);

/*
  float* pOut = new float[gridSize*5];
  hipMemcpy(pOut, pTemp, gridSize*5*sizeof(float), hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  
  for (int i = 0; i < gridSize; i++) {
    sum += pOut[i];
    sum2 += pOut[i + gridSize];
    sum3 = std::max<float>(sum3, pOut[i + 2*gridSize]);
    sum4 += pOut[i + 3*gridSize];
  }
  delete[] pOut;
  hipFree(pTemp);
  if(sum==0)
    sum=1;
  *maxval = sum3;
  *mean = sum4 / n;
  return sqrt(sum2 / sum); // - (sum / n) * (sum / n);
*/  
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
