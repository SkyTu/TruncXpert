// Author: Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "utils/gpu_mem.h"
#include "utils/misc_utils.h"
#include "utils/gpu_random.h"

#include "gpu_fss_helper.h"
#include "gpu_select.h"
#include "gpu_linear_helper.h"

// select(b, x-p, 0) + q
template <typename TIn, typename TOut, u64 p, u64 q>
__global__ void selectKernel(u32 *X,
                             TIn *Y,
                             TOut *a, TOut *b,
                             TOut *c, TOut *d1,
                             TOut *d2, int party, int N, int bw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        int laneId = threadIdx.x & 0x1f;
        TOut x = ((X[i / 32] >> laneId) & 1ULL);
        TOut is_zero_x = (x == 0);
        auto y = TOut(Y[i] - p);

        // y -= p;
        // gpuMod(y, bw);
        a[i] = -a[i] * y - b[i] * x + c[i] + y * is_zero_x * d1[i] +
               is_zero_x * d2[i] + (party == SERVER1) * (x * y + TOut(q));
        gpuMod(a[i], bw);
        // if (i < 8)
        // printf("select %d: %ld, %ld, %ld\n", i, i64(x), i64(y), i64(a[i]));
        // auto selectOutput = select<p>(drelu, diff, a, b, c, d1, d2, party, N, i);
        // if(party == SERVER1 && curMax != NULL) selectOutput += curMax[i];
        // a[i] = selectOutput;
    }
}

template <typename TIn, typename TOut, u64 p, u64 q>
TOut *gpuSelect(SigmaPeer *peer, int party, int bw, GPUSelectKey<TOut> k, u32 *d_x, TIn *d_Y, Stats *s, bool opMasked = true)
{
    assert(bw <= 8 * sizeof(TOut));
    size_t memSz = k.N * sizeof(TOut);

    TOut *d_a = (TOut *)moveToGPU((uint8_t *)k.a, memSz, s);
    TOut *d_b = (TOut *)moveToGPU((uint8_t *)k.b, memSz, s);
    TOut *d_c = (TOut *)moveToGPU((uint8_t *)k.c, memSz, s);
    TOut *d_d1 = (TOut *)moveToGPU((uint8_t *)k.d1, memSz, s);
    TOut *d_d2 = (TOut *)moveToGPU((uint8_t *)k.d2, memSz, s);
    // printf("Doing select\n");
    selectKernel<TIn, TOut, p, q><<<(k.N - 1) / 256 + 1, 256>>>(d_x, d_Y, d_a, d_b, d_c, d_d1, d_d2, party, k.N, bw);
    checkCudaErrors(cudaDeviceSynchronize());
    // printf("finished kernel\n");
    if (opMasked)
        peer->reconstructInPlace(d_a, bw, k.N, s);

    // gpuFree(d_a);
    gpuFree(d_b);
    gpuFree(d_c);
    gpuFree(d_d1);
    gpuFree(d_d2);

    return d_a;
}

template <typename TIn, typename TOut, typename TMaskB>
__global__ void keyGenSelectKernel(int N, TMaskB *maskB, TIn *maskX, TOut *randomMaskOut, TOut *maskOut, TOut *oneBitDcfK1, TOut *oneBitDcfK2, int bw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        // if (i == 0)
        //     printf("select key random mask %ld %ld %d\n", maskB[i], maskX[i], bw);
        maskOut[i] = TOut(maskB[i] * maskX[i]) + (randomMaskOut ? randomMaskOut[i] : 0);
        gpuMod(maskOut[i], bw);
        oneBitDcfK1[i] = 0;
        oneBitDcfK2[i] = 0;
        if (maskB[i] == TIn(1))
        {
            oneBitDcfK1[i] = 2;
            oneBitDcfK2[i] = -2 * maskX[i];
            gpuMod(oneBitDcfK1[i], bw);
            gpuMod(oneBitDcfK2[i], bw);
        }
    }
}

// if you don't have a random mask then the function returns one else it returns null
template <typename TIn, typename TOut, typename TMaskB>
TOut *gpuKeyGenSelect(uint8_t **key_as_bytes, int party, int N, TIn *d_maskX, TMaskB *d_maskB, int bw, bool opMasked = true)
{
    // printf("bw=%d, Tout=%d\n", bw, sizeof(TOut));
    assert(bw <= 8 * sizeof(TOut));
    if (!d_maskX)
        d_maskX = randomGEOnGpu<TIn>(N, bw);
    TOut *d_randomMaskOut = opMasked ? randomGEOnGpu<TOut>(N, bw) : NULL;
    // if (d_randomMaskOut)
    // {
    //     checkCudaErrors(cudaMemset(d_randomMaskOut, 0, N * sizeof(TOut)));
    // }
    auto d_out = (TOut *)gpuMalloc(N * sizeof(TOut));
    auto d_oneBitK1 = (TOut *)gpuMalloc(N * sizeof(TOut));
    auto d_oneBitK2 = (TOut *)gpuMalloc(N * sizeof(TOut));
    // printf("Bw=%d\n", bw);
    keyGenSelectKernel<<<(N - 1) / 256 + 1, 256>>>(N, d_maskB, d_maskX, d_randomMaskOut, d_out, d_oneBitK1, d_oneBitK2, bw);
    checkCudaErrors(cudaDeviceSynchronize());
    writeShares<TMaskB, TOut>(key_as_bytes, party, N, d_maskB, bw);
    writeShares<TIn, TOut>(key_as_bytes, party, N, d_maskX, bw);
    writeShares<TOut, TOut>(key_as_bytes, party, N, d_out, bw);
    writeShares<TOut, TOut>(key_as_bytes, party, N, d_oneBitK1, bw);
    writeShares<TOut, TOut>(key_as_bytes, party, N, d_oneBitK2, bw);
    gpuFree(d_out);
    gpuFree(d_oneBitK1);
    gpuFree(d_oneBitK2);
    return d_randomMaskOut;
}

template <typename T, typename TMaskB>
__global__ void keyGenSelectExtendKernel(T* inputMask, T* outputMask, TMaskB* rs, T* re, T* v, T* p, T* q, int bin, int bout, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        assert(rs[i] == 0 || rs[i] == 1);
        auto rmsb = gpuMsb(inputMask[i], bin);
        v[i] = (1 - rs[i]) * inputMask[i] - outputMask[i];
        gpuMod(v[i], bout);
        p[i] = rs[i] * rmsb;
        q[i] = (1 - rs[i]) * rmsb;
        re[i] = inputMask[i] - outputMask[i] - outputMask[i];
        gpuMod(re[i], bout);
    }
}

template <typename T>
__global__ void genSelectExtKernel(T* inputMask, T* outputMask, u8* rs, T* re, T* v, T* p, T* q, int bin, int bout, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        assert(rs[i] == 0 || rs[i] == 1);
        auto rmsb = gpuMsb(inputMask[i], bin);
        v[i] = (1 - rs[i]) * inputMask[i] - outputMask[i];
        gpuMod(v[i], bout);
        p[i] = rs[i] * rmsb;
        q[i] = (1 - rs[i]) * rmsb;
        re[i] = inputMask[i] - outputMask[i] - outputMask[i];
        gpuMod(re[i], bout);
    }
}

template <typename T>
T* gpuKeyGenSelectExt(uint8_t** key_as_bytes, int party, int bin, int bout, int N, u8* rs, T* inputMask){
    T* outputMask = randomGEOnGpu<T>(N, bout);
    T* re = (T*)gpuMalloc(N * sizeof(T));
    T* v = (T*)gpuMalloc(N * sizeof(T));
    T* p = (T*)gpuMalloc(N * sizeof(T));
    T* q = (T*)gpuMalloc(N * sizeof(T));
    
    genSelectExtKernel<<<(N - 1) / 256 + 1, 256>>>(inputMask, outputMask, rs, re, v, p, q, bin, bout, N);
    writeShares<T, T>(key_as_bytes, party, N, re, bout);
    writeShares<u8, T>(key_as_bytes, party, N, rs, bout);
    writeShares<T, T>(key_as_bytes, party, N, v, bout);
    writeShares<T, T>(key_as_bytes, party, N, p, bout);
    writeShares<T, T>(key_as_bytes, party, N, q, bout);
    
    gpuFree(v);
    gpuFree(p);
    gpuFree(q);
    gpuFree(re);
    return outputMask;
}


template <typename T, u64 p, u64 q>
__global__ void selectExtendKernel(int party, int bin, int bout, int N, T* d_I, u32* d_dcf, T* d_re, T* d_rs, T* d_v, T* d_p, T* d_q, T* res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        d_I[i] = d_I[i] + (1ULL << (bin - 2));
        gpuMod(d_I[i], bin);
        auto t = (1ULL - gpuMsb(d_I[i], bin)) * (1ULL << bin);
        int laneId = threadIdx.x & 0x1f;
        auto dhat = ((d_dcf[i / 32] >> laneId) & 1ULL);
        d_I[i] = d_I[i] - (1ULL << (bin - 2));
        gpuMod(d_I[i], bout);
        assert(dhat == 0 || dhat == 1);
        
        if(dhat){
            res[i] = (T(party) - d_rs[i]) * d_I[i] + t * (d_q[i]) - d_v[i];
        }
        else{
            res[i] = d_rs[i] * d_I[i] + t * d_p[i] + d_v[i] - d_re[i];
        }
        gpuMod(res[i], bout);
    }
}

template <typename T, u64 p, u64 q>
T *gpuSelectExtend(SigmaPeer *peer, int bin, int bout, int party, GPUSelectExtKey<T> k, u32 *d_dcf,T *d_I, Stats *s, bool opMasked = true)
{
    // d_x 是d_drelu, d_Y是incoming grad
    assert(bin <= 8 * sizeof(T));
    assert(bout <= 8 * sizeof(T));
    size_t memSz = k.N * sizeof(T);
    auto d_re = (T *)moveToGPU((uint8_t *)k.re, k.N * sizeof(T), s);
    auto d_rs = (T *)moveToGPU((uint8_t *)k.rs, k.N * sizeof(T), s);
    auto d_v = (T *)moveToGPU((uint8_t *)k.v, k.N * sizeof(T), s);
    auto d_p = (T *)moveToGPU((uint8_t *)k.p, k.N * sizeof(T), s);
    auto d_q = (T *)moveToGPU((uint8_t *)k.q, k.N * sizeof(T), s);
    T *d_out = (T *)gpuMalloc(memSz);
    // printf("Doing select\n");
    selectExtendKernel<T, p, q><<<(k.N - 1) / 256 + 1, 256>>>(party, bin, bout, k.N, d_I, d_dcf, d_re, d_rs, d_v, d_p, d_q, d_out);
    checkCudaErrors(cudaDeviceSynchronize());
    // printf("finished kernel\n");
    if (opMasked)
        peer->reconstructInPlace(d_out, bout, k.N, s);
    gpuFree(d_re);
    gpuFree(d_v);
    gpuFree(d_p);
    gpuFree(d_q);
    return d_out;
}