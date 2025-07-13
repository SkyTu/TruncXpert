#include <stdio.h>
#include <cmath>
#include <cassert>
#include <cstdint>

#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "fss/wing/gpu_truncate.h"
#include "fss/wing/gpu_relu.h"

using T = u64;

inline T cpuMsb(T x, int bin){
    return ((x >> (bin - 1)) & T(1));
}

int global_device = 0;

int main(int argc, char *argv[]) {
    int party = atoi(argv[1]);
    auto peer = new GpuPeer(true);
    peer->connect(party, argv[2]);
    int N = atoi(argv[3]);
    global_device = atoi(argv[4]);
    AESGlobalContext g;
    Stats stats; // 在栈上创建一个 Stats 对象
    Stats* S = &stats; // 让指针 S 指向 stats
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
    size_t keyBufSz = 8 * OneGB;
    getKeyBuf(&startPtr, &curPtr, keyBufSz);
    T* h_r = (T*) cpuMalloc(N * sizeof(T));
    wing::TruncateType t = wing::TruncateType::StochasticTR;

    // generate TReKey
    auto d_truncateMask = wing::genGPUTruncateKey(&curPtr, party, t, bin, bout, shift, N, d_mask, &g, h_r);
    assert(curPtr - startPtr < keyBufSz);
    printf("Key size=%lu\n", curPtr - startPtr);
    auto d_tempMask = wing::gpuKeyGenReluZeroExt(&curPtr, party, bin-shift, bout, N, d_truncateMask, &g);
    auto d_dreluMask = d_tempMask.first;
    gpuFree(d_dreluMask);
    auto d_reluMask = d_tempMask.second;
    assert(curPtr - startPtr < keyBufSz);
    printf("Key size=%lu\n", curPtr - startPtr);
    auto h_mask_O = (T*) moveToCPU((u8*) d_reluMask, N * sizeof(T), NULL);
    
    curPtr = startPtr;
    std::cout << "Reading key\n";
    auto k = wing::readGPUTruncateKey<T>(t, &curPtr);
    auto k1 = wing::readGPUReluZeroExtKey<T>(&curPtr);

    auto start_send = peer->peer->keyBuf->bytesSent;
    auto start = std::chrono::high_resolution_clock::now();
    wing::gpuTruncate(bin, bout, t, k, shift, peer, party, N, d_X_share, &g, S, true);
    auto temp = wing::gpuReluZeroExt(peer, party, k1, d_X_share, &g, (Stats *)NULL, false);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    auto end_send = peer->peer->keyBuf->bytesSent;
    std::cout << "Send " << end_send - start_send << " bytes." << std::endl;

    auto d_drelu = temp.first;
    gpuFree(d_drelu);
    T *d_relu;
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