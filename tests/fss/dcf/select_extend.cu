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

#include <cassert>
#include <cstdint>
#include <sytorch/tensor.h>

#include "../../../utils/gpu_data_types.h"
#include "../../../utils/gpu_file_utils.h"
#include "../../../utils/misc_utils.h"
#include "../../../utils/gpu_mem.h"
#include "../../../utils/gpu_random.h"
#include "../../../utils/gpu_comms.h"

#include "../../../fss/dcf/gpu_relu.h"
#include "../../../fss/dcf/gpu_truncate.h"

using T = u64;
int global_device = 0;
int main(int argc, char *argv[]) {
    
    int party = atoi(argv[1]);
    
    global_device = atoi(argv[4]);
    auto peer = new GpuPeer(true);
    peer->connect(party, argv[2]);
    int N = atoi(argv[3]);
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    // initCommBufs(true);
    int bin = 64;
    int bout = 64;
    int shift = 24;
    
    std::cout << "generate random numbers" << std::endl;
    // generate the share of x + rin
    auto h_X = new T[N];
    auto d_X = randomGEOnGpuWithGap<T>(N, bin, 2);
    // generate rin
    auto d_mask = randomGEOnGpu<T>(N, bin);
    // generate x = x_0 + x_1 - rin
    auto d_masked_X = (T*) gpuMalloc(N * sizeof(T));
    gpuLinearComb(64, N, d_masked_X, T(1), d_X, T(1), d_mask);
    auto d_X_share = randomGEOnGpu<T>(N, bin);
    if (party == 1)
        gpuLinearComb(64, N, d_X_share, T(1), d_masked_X, T(-1), d_X_share);
    h_X = (T *)moveToCPU((u8 *)d_X, N * sizeof(T), NULL); 

    u8 *startPtr, *curPtr;
    size_t keyBufSz = 10 * OneGB;
    getKeyBuf(&startPtr, &curPtr, keyBufSz);
    
    std::cout << "begin generate masks" << std::endl;
    auto reluRes = new u8[N];
    u8 * dcf_mask = new u8[N];
    // generate fake mask
    for(int i = 0; i < N; i++){
        dcf_mask[i] = 0; // fake mask
    }

    // generate masked dcfRes
    for(int i = 0; i < N; i++){
        reluRes[i] = 0; // initialize ReLU result
    }

    std::cout << "generate reluRes" << std::endl;
    if(party == 1){
        for(int i = 0; i < N; i++){
            if((h_X[i] < (1ULL << (bin - 1)))){
                reluRes[i] = 1;
                if(i < 100)
                    printf("%d: x is %ld, reluRes is %d\n", i, h_X[i], reluRes[i]);
            }
            else{
                reluRes[i] = 0;
                if(i < 100)
                    printf("%d: x is %ld, reluRes is %d\n", i, h_X[i], reluRes[i]);
            }
        }
    }

    auto d_dcfMask = (u8*) moveToGPU((u8*) dcf_mask, N * sizeof(u8), NULL);

    std::cout << "Finish Secret Share" << std::endl;
    auto d_dcf = (u8*) moveToGPU((u8*) reluRes, N * sizeof(u8), NULL);
    peer->reconstructInPlace((u8*)d_dcf, 1, N, NULL);
    auto memSz = size_t(((u64)N - 1) / 64 + 1) * 8;
    auto d_compresseddcf = (u8 *)gpuMalloc(memSz);
    u64 threads = memSz / 8; //(memSz - 1) / 8 + 1;
    // printf("%lu\n", threads);
    compressKernel<<<(threads - 1) / 128 + 1, 128>>>(1, 1, threads, N, d_dcf, d_compresseddcf);
    std::cout << "Begin Generation" << std::endl;
    // Note that in Orca, the backward propagation first invoke truncate, then invoke select
    // generate Truncate Key
    dcf::TruncateType t = dcf::TruncateType::StochasticTruncate;

    auto d_outputMask = dcf::genGPUTruncateKey(&curPtr, party, t, bin, bout, shift, N, d_mask, &g);
    // generate Select Key for backward propagation
    auto d_outMask = gpuKeyGenSelect<u64, u64, u8>(&curPtr, party, N, d_outputMask, d_dcfMask, bout);
    auto h_outMask = (T*) moveToCPU((u8*) d_outMask, N * sizeof(T), NULL);
    printf("Key size=%lu\n", curPtr - startPtr);
    curPtr = startPtr;
    // read truncate key
    auto k0 = dcf::readGPUTruncateKey<T>(t, &curPtr);
    // read select key
    auto k1 = readGPUSelectKey<T>(&curPtr, N);
    
    peer->sync();
    auto start_send = peer->peer->keyBuf->bytesSent;
    auto start = std::chrono::high_resolution_clock::now();
    peer->reconstructInPlace((u64*)d_X_share, bin, N, NULL);
    dcf::gpuTruncate(bin, bout, t, k0, shift, peer, party, N, d_X_share, &g, NULL);
    auto d_selectOutput = gpuSelect<T, T, 0, 0>(peer, party, bout, k1, (u32 *)d_compresseddcf, d_X_share, NULL, false);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    auto end_send = peer->peer->keyBuf->bytesSent;
    std::cout << "Send " << end_send - start_send << " bytes." << std::endl;

    peer->reconstructInPlace((u64*)d_selectOutput, bin, N, NULL);
    auto h_O = (T*)moveToCPU((u8*) d_selectOutput, N * sizeof(T), NULL);
    // 计算结果是存在d_mask_X的
    destroyGPURandomness();

    for (int i = 0; i < N; i++)
    {
        auto unmasked_O = h_O[i] - h_outMask[i];
        cpuMod(unmasked_O, bout);
        auto o = (h_X[i] < (1ULL << (bin - 1))) * (h_X[i] >> shift);
        if (i < 100)
            printf("%d: x is %ld, output is %ld, expected is %ld\n", i, h_X[i], unmasked_O, o);
    }
    // checkTrStochastic(bin, bout, shift, N, h_O, h_outputMask, h_I, h_r);
}