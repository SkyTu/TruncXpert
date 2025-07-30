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

#include "../../../utils/gpu_data_types.h"
#include "../../../utils/gpu_file_utils.h"
#include "../../../utils/misc_utils.h"
#include "../../../utils/gpu_mem.h"
#include "../../../utils/gpu_random.h"
#include "../../../utils/gpu_comms.h"

#include "../../../fss/dcf/gpu_relu.h"
#include "../../../fss/dcf/gpu_truncate.h"

#include <cassert>
#include <sytorch/tensor.h>

using T = u64;
int global_device = 0;
double wan_time = 0;
int main(int argc, char *argv[])
{
    global_device = atoi(argv[4]);
    // initCommBufs(true);
    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    int bin = 64;
    int bout = 64;
    int shift = 24;
    int N = atoi(argv[3]); //8;
    int party = atoi(argv[1]);

    auto peer = new GpuPeer(true);
    peer->connect(party, argv[2]);

    u8 *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 10 * OneGB);
    initGPURandomness();
    
    // generate the share of x + rin
    auto h_X = new T[N];
    auto d_X = randomGEOnGpuWithGap<T>(N, bin, 2);
    // generate rin
    auto d_mask_X = randomGEOnGpu<T>(N, bin);
    auto h_mask_X = (T *)moveToCPU((u8 *)d_mask_X, N * sizeof(T), NULL);
    // generate x = x_0 + x_1 - rin
    auto d_masked_X = (T*) gpuMalloc(N * sizeof(T));
    gpuLinearComb(bin, N, d_masked_X, T(1), d_X, T(1), d_mask_X);
    auto d_X_share = randomGEOnGpu<T>(N, bin);
    if (party == 1)
        gpuLinearComb(bin, N, d_X_share, T(1), d_masked_X, T(-1), d_X_share);
    h_X = (T *)moveToCPU((u8 *)d_X, N * sizeof(T), NULL); 
    
    dcf::TruncateType t = dcf::TruncateType::StochasticTR;
    auto d_outputMask = dcf::genGPUTruncateKey(&curPtr, party, t, bin, bout, shift, N, d_mask_X, &g);
    auto d_tempMask = dcf::gpuKeygenReluExtend(&curPtr, party, bin-shift, bout, N, d_outputMask, &g);
    auto d_dreluMask = d_tempMask.first;
    gpuFree(d_dreluMask);
    auto d_reluMask = d_tempMask.second;
    
    auto h_mask_O = (T *)moveToCPU((u8 *)d_reluMask, N * sizeof(T), NULL);
    auto k0 = dcf::readGPUTruncateKey<T>(t, &startPtr);
    auto k1 = dcf::readGPUReluExtendKey<T>(&startPtr);
    T *d_relu;
    
    
    peer->sync();
    auto start_send = peer->peer->keyBuf->bytesSent;
    auto start = std::chrono::high_resolution_clock::now();
    peer->reconstructInPlace((u64*)d_X_share, bin, N, NULL);
    dcf::gpuTruncate(bin, bout, t, k0, shift, peer, party, N, d_X_share, &g, NULL);
    auto temp = dcf::gpuReluExtend(peer, party, k1, d_X_share, &g, (Stats *)NULL, false);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    auto end_send = peer->peer->keyBuf->bytesSent;
    std::cout << "Send " << end_send - start_send << " bytes." << std::endl;
    std::cout << "Wan time: "  << wan_time << " ms" << std::endl;

    auto d_drelu = temp.first;
    gpuFree(d_drelu);
    d_relu = temp.second;
    peer->reconstructInPlace((u64*)d_relu, bout, N, NULL);
    auto h_relu = (T *)moveToCPU((u8 *)d_relu, N * sizeof(T), (Stats *)NULL);
    gpuFree(d_relu);
    destroyGPURandomness();

    for (int i = 0; i < N; i++)
    {
        auto unmasked_O = (h_relu[i] - h_mask_O[i]);
        cpuMod(unmasked_O, bout);
        auto o = (h_X[i] < (1ULL << (bin - 1))) * (h_X[i] >> shift);
        if (i < 100)
            printf("%d: x is %ld, output is %ld, expected is %ld\n", i, h_X[i], unmasked_O, o);
        // assert(o == unmasked_O);
    }

    return 0;
}