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

#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "fss/dcf/gpu_truncate.h"

using T = u64;
int global_device = 0;
double wan_time = 0;
int main(int argc, char *argv[]) {
    int N = atoi(argv[3]);
    int party = atoi(argv[1]);
    
    auto peer = new GpuPeer(true);
    peer->connect(party, argv[2]);
    global_device = atoi(argv[4]);
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    // initCommBufs(true);
    int bin = 64;
    int bout = 64;
    int shift = 24;
    

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
    T* h_r = (T*) cpuMalloc(N * sizeof(T));
    auto d_outputMask = dcf::genGPUStochasticTruncateKey(&curPtr, party, bin, bout, shift, N, d_mask, &g, h_r);
    assert(curPtr - startPtr < keyBufSz);
    auto h_outputMask = (T*) moveToCPU((u8*) d_outputMask, N * sizeof(T), NULL);
    gpuFree(d_outputMask);

    curPtr = startPtr;
    auto k = dcf::readGPUTrStochasticKey<T>(&curPtr);
    printf("Key size=%lu\n", curPtr - startPtr);

    auto start_send = peer->peer->keyBuf->bytesSent;
    auto start = std::chrono::high_resolution_clock::now();
    peer->reconstructInPlace((u64*)d_X_share, bin, N, NULL);
    dcf::gpuStochasticTruncate(k, party, peer, d_X_share, &g, (Stats*) NULL, false);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    auto end_send = peer->peer->keyBuf->bytesSent;
    std::cout << "Send " << end_send - start_send << " bytes." << std::endl;
    std::cout << "Wan time: " << wan_time << std::endl;
    peer->reconstructInPlace((u64*)d_X_share, bin, N, NULL);
    auto h_O = (T*) moveToCPU((u8*) d_X_share, N * sizeof(T), NULL);
    // 计算结果是存在d_mask_X的
    destroyGPURandomness();

    for (int i = 0; i < N; i++)
    {
        auto unmasked_O = h_O[i] - h_outputMask[i];
        cpuMod(unmasked_O, bout);
        auto o = cpuArs(h_X[i], bin, shift);
        cpuMod(o, bout);
        if (o != unmasked_O){
            if (int(o-unmasked_O) > 1 || int(o-unmasked_O) < -1){
                printf("%d: h_x = %ld, real_truncate = %ld, stTR_res = %ld, diff = %ld\n", i, h_X[i], o, unmasked_O, o-unmasked_O);
            }
        }
    }
    // checkTrStochastic(bin, bout, shift, N, h_O, h_outputMask, h_I, h_r);
}