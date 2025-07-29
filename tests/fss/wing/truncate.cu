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

using T = u64;

inline T cpuMsb(T x, int bin){
    return ((x >> (bin - 1)) & T(1));
}
double wan_time = 0;
int global_device = 0;

int main(int argc, char *argv[]) {
    int party = atoi(argv[1]);
    auto peer = new GpuPeer(true);
    peer->connect(party, argv[2], atoi(argv[3]));
    int N = atoi(argv[4]);
    global_device = atoi(argv[5]);
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
    size_t keyBufSz = 10 * OneGB;
    getKeyBuf(&startPtr, &curPtr, keyBufSz);
    T* h_r = (T*) cpuMalloc(N * sizeof(T));
    wing::TruncateType t = wing::TruncateType::StochasticTruncate;

    // generate TReKey
    auto d_truncateMask = wing::genGPUTruncateKey(&curPtr, party, t, bin, bout, shift, N, d_mask, &g, h_r);
    assert(curPtr - startPtr < keyBufSz);
    printf("Key size=%lu\n", curPtr - startPtr);
    auto h_truncateMask = (T*) moveToCPU((u8*) d_truncateMask, N * sizeof(T), NULL);
    
    curPtr = startPtr;
    std::cout << "Reading key\n";
    auto k = wing::readGPUTruncateKey<T>(t, &curPtr);
    
    auto start_send = peer->peer->keyBuf->bytesSent;
    auto start = std::chrono::high_resolution_clock::now();
    wing::gpuTruncate(bin, bout, t, k, shift, peer, party, N, d_X_share, &g, S, false);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    std::cout << "Time taken" << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "micros\n";
    auto end_send = peer->peer->keyBuf->bytesSent;
    std::cout << "Send " << end_send - start_send << " bytes." << std::endl;
    std::cout << "Wan time: " << wan_time << std::endl;

    auto h_TRe = new T[N];
    peer->reconstructInPlace((u64*)d_X_share, bout, N, NULL);
    h_TRe = (T*) moveToCPU((u8*) d_X_share, N * sizeof(T), NULL);
    // 计算结果是存在d_mask_X的
    destroyGPURandomness();
    printf("Verify!");
    for (int i = 0; i < N; i++)
    {
        auto unmasked_TRe = h_TRe[i] - h_truncateMask[i];
        cpuMod(unmasked_TRe, bout);
        auto o = cpuArs(h_X[i], bin, shift);
        cpuMod(o, bout);
        if (o != unmasked_TRe){
            if (int(o-unmasked_TRe) > 1 || int(o-unmasked_TRe) < -1){
                printf("%d: h_x = %ld, real_truncate = %ld, stTR_res = %ld, diff = %ld\n", i, h_X[i], o, unmasked_TRe, o-unmasked_TRe);
            }
        }
    }
}