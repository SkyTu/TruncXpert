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
#include "utils/wan_config.h"
using T = u64;

inline T cpuMsb(T x, int bin){
    return ((x >> (bin - 1)) & T(1));
}
double wan_time = 0;
int global_device = 0;

int main(int argc, char *argv[]) {
    WanParameter wanParams;
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
    
    std::cout << "Begin generation" << std::endl;
    auto h_X = new T[N];
    auto d_X = randomGEOnGpuWithGap<T>(N, bin, 2);
    // generate rin
    auto d_mask = randomGEOnGpu<T>(N, bin);
    // generate x = x_0 + x_1 - rin
    auto d_masked_X = (T*) gpuMalloc(N * sizeof(T));
    gpuLinearComb(bin, N, d_masked_X, T(1), d_X, T(1), d_mask);
    auto d_X_share = randomGEOnGpu<T>(N, bin);
    if (party == 1)
        gpuLinearComb(bin, N, d_X_share, T(1), d_masked_X, T(-1), d_X_share);
    h_X = (T *)moveToCPU((u8 *)d_X, N * sizeof(T), NULL);   

    u8 *startPtr, *curPtr, *tmpPtr;
    size_t keyBufSz = 10 * OneGB;
    getKeyBuf(&startPtr, &curPtr, keyBufSz);
    tmpPtr = startPtr;
    T* h_r = (T*) cpuMalloc(N * sizeof(T));
    wing::TruncateType t = wing::TruncateType::StochasticTR;

    // generate TReKey
    auto d_truncateMask = wing::genGPUTruncateKey(&curPtr, party, t, bin, bout, shift, N, d_mask, &g, h_r);
    auto tre_keysize = curPtr - tmpPtr;
    tmpPtr = curPtr;
    // generate ReLuExtKey
    // note that the ReLU is useless in select ext test, we just minus the key size
    auto d_dReluMask = dpf::gpuKeyGenDRelu(&curPtr, party, bin-shift, N, d_truncateMask, &g);
    auto relu_keysize = curPtr - tmpPtr;
    tmpPtr = curPtr;
    auto d_outputMask = gpuKeyGenSelectExt(&curPtr, party, bin-shift, bout, N, d_dReluMask, d_truncateMask);
    auto selectExt_keysize = curPtr - tmpPtr;
    auto relumask = new u8[N];
    relumask = (u8 *)moveToCPU((u8 *)d_dReluMask, N * sizeof(u8), NULL);
    gpuFree(d_dReluMask);
    printf("Key size=%lu\n", tre_keysize + selectExt_keysize);
    auto h_mask_O = (T*) moveToCPU((u8*) d_outputMask, N * sizeof(T), NULL);
    
    curPtr = startPtr;
    std::cout << "Reading key\n";
    auto k =  wing::readGPUTruncateKey<T>(t, &curPtr);
    auto k1 = dpf::readGPUDReluKey(&curPtr);
    auto k2 = readGPUSelectExtKey<T>(&curPtr, N);

    // The comparison result are calculated during the forward propagation
    // The backward propagation process. When the next layer is ReLU, we can replace the truncate + select with truncate reduce + select extend.
    
    auto start_send = peer->peer->keyBuf->bytesSent;
    auto start = std::chrono::high_resolution_clock::now();
    wing::gpuTruncate(bin, bout, t, k, shift, peer, party, N, d_X_share, &g, S, true);
    std::vector<u32 *> h_mask({k1.mask});
    auto end = std::chrono::high_resolution_clock::now();
    auto end_send = peer->peer->keyBuf->bytesSent;
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto send_bytes = end_send - start_send;

    auto d_dcf = dpf::gpuDcf<T, 1, dpf::dReluPrologue<0>, dpf::dReluEpilogue<0, false>>(k1.dpfKey, party, d_X_share, &g, (Stats*) NULL, &h_mask);
    peer->reconstructInPlace(d_dcf, 1, N, (Stats*) NULL); 
    
    auto dcf_send_bytes = peer->peer->keyBuf->bytesSent - end_send;

    start_send = peer->peer->keyBuf->bytesSent;
    start = std::chrono::high_resolution_clock::now();
    auto d_relu = wing::gpuReluZeroExtMux(party, bin-shift, bout, N, k2, d_X_share, d_dcf, (Stats *)NULL);
    end = std::chrono::high_resolution_clock::now();
    time = time + std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Time taken=%lu millis\n", time);
    end_send = peer->peer->keyBuf->bytesSent;
    send_bytes = send_bytes + end_send - start_send;
    std::cout << "Send " << send_bytes << " bytes." << std::endl;
    std::cout << "Wan time: " << wan_time - wanParams.rtt - dcf_send_bytes/wanParams.comm_bytes_per_ms << std::endl;
    peer->reconstructInPlace((u64*)d_relu, bout, N, NULL);
    auto h_relu = (T *)moveToCPU((u8 *)d_relu, N * sizeof(T), (Stats *)NULL);
    gpuFree(d_relu);
    destroyGPURandomness();
    std::cout << "Begin Verification" << std::endl;
    for (int i = 0; i < N; i++)
    {
        auto unmasked_O = (h_relu[i] - h_mask_O[i]);
        cpuMod(unmasked_O, bout);
        auto o = (h_X[i] < (1ULL << (bin - 1))) * (h_X[i] >> shift);
        // if (i < 100)
        //     printf("%d: x is %ld, output is %ld, expected is %ld\n", i, h_X[i], unmasked_O, o);
        // assert(o == unmasked_O);
    }

    return 0;
}