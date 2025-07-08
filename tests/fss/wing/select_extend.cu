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
    int bin = 40;
    int bout = 64;
    
    // generate the share of x + rin
    auto h_X = new T[N];
    auto d_X = randomGEOnGpuWithGap<T>(N, bin, 2);
    // generate rin
    auto d_mask = randomGEOnGpu<T>(N, bin);
    // generate x = x_0 + x_1 - rin
    auto d_masked_X = (T*) gpuMalloc(N * sizeof(T));
    gpuLinearComb(64, N, d_masked_X, T(1), d_X, T(1), d_mask);
    h_X = (T *)moveToCPU((u8 *)d_X, N * sizeof(T), NULL);    
    

    u8 *startPtr, *curPtr;
    size_t keyBufSz = 10 * OneGB;
    getKeyBuf(&startPtr, &curPtr, keyBufSz);
    T* h_r = (T*) cpuMalloc(N * sizeof(T));

    // generate TReKey
    auto d_tempMask = wing::gpuKeyGenReluZeroExt(&curPtr, party, bin, bout, N, d_mask, &g);
    auto d_dreluMask = d_tempMask.first;
    gpuFree(d_dreluMask);
    auto d_reluMask = d_tempMask.second;
    assert(curPtr - startPtr < keyBufSz);
    printf("Key size=%lu\n", curPtr - startPtr);
    auto h_mask_O = (T*) moveToCPU((u8*) d_reluMask, N * sizeof(T), NULL);
    
    curPtr = startPtr;
    std::cout << "Reading key\n";
    auto k1 = dpf::readGPUDReluKey(&curPtr);
    auto k2 = wing::readGPUSelectExtKey<T>(&curPtr, N);

    // The comparison result and d_masked_X are calculated during the forward propagation
    std::vector<u32 *> h_mask({k1.mask});
    auto d_dcf = dpf::gpuDcf<T, 1, dpf::dReluPrologue<0>, dpf::dReluEpilogue<0, false>>(k1.dpfKey, party, d_masked_X, &g, (Stats*) NULL, &h_mask);
    peer->reconstructInPlace(d_dcf, 1, N, (Stats*) NULL); 
    auto start_send = peer->peer->keyBuf->bytesSent;
    auto start = std::chrono::high_resolution_clock::now();
    auto d_relu = wing::gpuReluZeroExtMux(party, bin, bout, N, k2, d_masked_X, d_dcf, (Stats *)NULL);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    auto end_send = peer->peer->keyBuf->bytesSent;
    std::cout << "Send " << end_send - start_send << " bytes." << std::endl;

    peer->reconstructInPlace((u64*)d_relu, bout, N, NULL);
    auto h_relu = (T *)moveToCPU((u8 *)d_relu, N * sizeof(T), (Stats *)NULL);
    gpuFree(d_relu);
    destroyGPURandomness();
    std::cout << "Begin Verification" << std::endl;
    int shift = bout-bin;
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