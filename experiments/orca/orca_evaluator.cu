// 
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
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <omp.h>
#include <string>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_random.h"
#include "../datasets/gpu_data.h"

#include "nn/orca/gpu_layer.h"
#include "nn/orca/gpu_model.h"

#include "cnn.h"
#include "model_accuracy.h"

#include <sytorch/softmax.h>
#include <sytorch/backend/llama_base.h>

#include "cuda_runtime_api.h"
#include "utils/wan_config.h"

u64 *gpuSoftmax(int batchSz, int numClasses, int party, SigmaPeer *peer, u64 *d_I, u64 *labels, bool secfloat, LlamaBase<u64> *llama)
{
    Tensor4D<u64> inp(batchSz, numClasses, 1, 1);
    Tensor4D<u64> softmaxOp(batchSz, numClasses, 1, 1);

    size_t memSz = batchSz * numClasses * sizeof(u64);
    moveIntoCPUMem((u8 *)inp.data, (u8 *)d_I, memSz, NULL);
    gpuFree(d_I);
    if (secfloat)
    {
        softmax_secfloat(inp, softmaxOp, dcf::orca::global::scale, LlamaConfig::party);
    }
    else
    {
        pirhana_softmax(inp, softmaxOp, dcf::orca::global::scale);
        // softmax<u64,dcf::orca::global::scale>(inp, softmaxOp);
    }
    for (int img = 0; img < batchSz; img++)
    {
        for (int c = 0; c < numClasses; c++)
        {
            softmaxOp(img, c, 0, 0) -= (labels[numClasses * img + c] * (((1LL << dcf::orca::global::scale)) / batchSz));
        }
    }
    reconstruct(inp.d1 * inp.d2, softmaxOp.data, 64);
    d_I = (u64 *)moveToGPU((u8 *)softmaxOp.data, memSz, NULL);
    return d_I;
}

void trainModel(dcf::orca::GPUModel<u64> *m, u8 **keyBuf, int party, SigmaPeer *peer, u64 *data, u64 *labels, AESGlobalContext *g, bool secfloat, LlamaBase<u64> *llama, int epoch, int iteration)
{
    auto start = std::chrono::high_resolution_clock::now();
    size_t inpMemSz = m->inpSz * sizeof(u64);
    auto d_I = (u64 *)moveToGPU((u8 *)data, inpMemSz, &(m->layers[0]->s));
    u64 *d_O;
    for (int i = 0; i < m->layers.size(); i++)
    {
        // std::cout << "Read Key Layer " << i << " begin" << std::endl;
        m->layers[i]->readForwardKey(keyBuf);
        // std::cout << "Layer " << i << " begin" << std::endl;
        d_O = m->layers[i]->forward(peer, party, d_I, g);
        // std::cout << "Layer " << i << " done" << std::endl;
        if (d_O != d_I)
            gpuFree(d_I);
        d_I = d_O;
    }
    checkCudaErrors(cudaDeviceSynchronize());
    d_I = gpuSoftmax(m->batchSz, m->classes, party, peer, d_I, labels, secfloat, llama);
    // std::cout << "Softmax finished" << std::endl;
    for (int i = m->layers.size() - 1; i >= 0; i--)
    {
        m->layers[i]->readBackwardKey(keyBuf, epoch);
        d_I = m->layers[i]->backward(peer, party, d_I, g, epoch);
        // std::cout << "Layer " << i << " backward done" << std::endl;
    }
}

void trainModelPerf(dcf::orca::GPUModel<u64> *m, u8 **keyBuf, int party, SigmaPeer *peer, u64 *data, u64 *labels, AESGlobalContext *g, bool secfloat, LlamaBase<u64> *llama, int epoch, int iteration, int & float_softmax_time)
{
    auto start = std::chrono::high_resolution_clock::now();
    size_t inpMemSz = m->inpSz * sizeof(u64);
    auto d_I = (u64 *)moveToGPU((u8 *)data, inpMemSz, &(m->layers[0]->s));
    u64 *d_O;
    for (int i = 0; i < m->layers.size(); i++)
    {
        // std::cout << "Read Key Layer " << i << " begin" << std::endl;
        m->layers[i]->readForwardKey(keyBuf);
        // std::cout << "Layer " << i << " begin" << std::endl;
        d_O = m->layers[i]->forward(peer, party, d_I, g);
        // std::cout << "Layer " << i << " done" << std::endl;
        if (d_O != d_I)
            gpuFree(d_I);
        d_I = d_O;
    }
    checkCudaErrors(cudaDeviceSynchronize());
    auto computeStart = std::chrono::high_resolution_clock::now();
    d_I = gpuSoftmax(m->batchSz, m->classes, party, peer, d_I, labels, secfloat, llama);
    auto computeEnd = std::chrono::high_resolution_clock::now();
    float_softmax_time += std::chrono::duration_cast<std::chrono::milliseconds>(computeEnd - computeStart).count();
    // std::cout << "Softmax finished" << std::endl;
    for (int i = m->layers.size() - 1; i >= 0; i--)
    {
        m->layers[i]->readBackwardKey(keyBuf, epoch);
        d_I = m->layers[i]->backward(peer, party, d_I, g, epoch);
        // std::cout << "Layer " << i << " backward done" << std::endl;
    }
}

u64 getKeySz(std::string dir, std::string modelName)
{
    std::ifstream kFile(dir + modelName + ".txt");
    u64 keySz;
    kFile >> keySz;
    return keySz;
}

void rmWeights(std::string lossDir, int party, int l, int k)
{
    assert(std::filesystem::remove(lossDir + "weights_mask_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + ".dat"));
    assert(std::filesystem::remove(lossDir + "masked_weights_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + ".dat"));
}

void evaluatorE2E(std::string modelName, std::string dataset, int party, std::string ip, std::string weightsFile, bool floatWeights, int epochs, int blocks, int blockSz, int batchSz, int H, int W, int C, bool secfloat, bool momentum, std::string keyDir, bool fake_offline = true)
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPUMemPool();
    initGPURandomness();
    initCPURandomness();
    // assert(epochs < 6);

    omp_set_num_threads(2);

    printf("Sync=%d\n", sync);
    printf("Opening fifos\n");
    char one = 1;
    char two = 2;

    auto trainingDir = "output/P" + std::to_string(party) + "/training/";
    auto expName = modelName + "-" + std::to_string(epochs) + "e-" + std::to_string(blocks) + "b";
    auto lossDir = trainingDir + "loss/" + expName + "/";
    auto weightsDir = lossDir + "weights/";
    auto keySzDir = trainingDir + "keysize/";
    std::ofstream lossFile(lossDir + "loss.txt");
    std::ofstream accFile(lossDir + "accuracy.txt");

    dcf::orca::GPUModel<u64> *m = getGPUModel<u64>(modelName, Tensor<u64>(nullptr, {(u64)batchSz, (u64)H, (u64)W, (u64)C}));
    m->setTrain(momentum);
    printf("Model created\n");
    m->initWeights(weightsFile, floatWeights);
    printf("Weights initialized\n");

    u8 *keyBuf1, *keyBuf2, *curKeyBuf, *nextKeyBuf;
    u64 keySz = getKeySz(keySzDir, modelName);
    getAlignedBuf(&keyBuf1, keySz);
    getAlignedBuf(&keyBuf2, keySz);
    int curBuf = 0;
    curKeyBuf = keyBuf1;
    nextKeyBuf = keyBuf2;

    SigmaPeer *peer = new GpuPeer(true);
    LlamaBase<u64> *llama = nullptr;

    // automatically truncates by scale
    LlamaConfig::party = party + 2;
    LlamaConfig::bitlength = dcf::orca::global::bw;
    llama = new LlamaBase<u64>();
    if (LlamaConfig::party == SERVER)
        llama->initServer(ip, (char **)&curKeyBuf);
    else
        llama->initClient(ip, (char **)&curKeyBuf);
    peer->peer = LlamaConfig::peer;

    if (secfloat)
        secfloat_init(party + 1, ip);
    
    std::string keyFile = keyDir + modelName + "_training_key" + std::to_string(party);
    // dropOSPageCache();
    std::chrono::duration<int64_t, std::milli> onlineTime = std::chrono::duration<int64_t, std::milli>::zero();
    std::chrono::duration<int64_t, std::milli> computeTime = std::chrono::duration<int64_t, std::milli>::zero();
    uint64_t keyReadTime = 0;
    size_t commBytes = 0;
    printf("Starting training\n");
    
    Dataset d = readDataset(dataset, party);
    int fd = openForReading(keyFile + "_" + to_string(0) + "_" + to_string(0) + "_" + std::to_string(0) + ".dat");
    for (int l = 0; l < epochs; l++)
    {
        for (int k = 0; k < blocks; k++)
        {
            initGPUMemPool();
            // Open the key file for reading
            // uncomment for end to end run
            peer->sync();
            auto startComm = peer->bytesReceived();
            auto start = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < blockSz; j++)
            {
                readKey(fd, keySz, curKeyBuf, &keyReadTime);
                peer->sync();
                auto computeStart = std::chrono::high_resolution_clock::now();
                auto labelsIdx = (k * blockSz + j) * batchSz * d.classes;
                int dataIdx = (k * blockSz + j) * d.H * d.W * d.C * batchSz;
                trainModel(m, &curKeyBuf, party, peer, &(d.data[dataIdx]), &(d.labels[labelsIdx]), &g, secfloat, llama, l, l * blocks * blockSz + k * blockSz + j);
                auto computeEnd = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(computeEnd - computeStart);
                computeTime += elapsed;  
                curKeyBuf = &keyBuf1[0]; 
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            onlineTime += elapsed;
            printf("Online time (ms): %lu\n", elapsed.count());
            auto endComm = peer->bytesReceived();
            commBytes += (endComm - startComm);
            std::pair<double, double> res;
            m->dumpWeights(weightsDir + "masked_weights_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + "_" + std::to_string(blockSz-1) + ".dat");
            if (dataset == "mnist")
            {
                std::cout << "Getting loss for MNIST" << std::endl;
                res = getLossMNIST<i64>(modelName, (u64)dcf::orca::global::scale, weightsDir, party, l, k, blockSz-1, true);
            }
            else
            {
                std::cout << "Getting loss for CIFAR10" << std::endl;
                res = getLossCIFAR10<i64>(modelName, (u64)dcf::orca::global::scale, weightsDir, party, l, k, blockSz-1, true);
            }
            auto accuracy = res.first;
            auto loss = res.second;
            printf("Accuracy=%lf, Loss=%lf\n", accuracy, loss);
            lossFile << loss << std::endl;
            accFile << accuracy << std::endl;   
        }
    }
    close(fd);


    LlamaConfig::peer->close();
    int iterations = epochs * blocks * blockSz;
    commBytes += secFloatComm;
    std::ofstream stats(trainingDir + expName + ".txt");
    auto statsString = "Total time taken (ms): " + std::to_string(onlineTime.count()) + "\nTotal bytes communicated: " + std::to_string(commBytes) + "\nSecfloat softmax bytes: " + std::to_string(inputOnlineComm + secFloatComm);

    auto avgKeyReadTime = (double)keyReadTime / (double)iterations;
    auto avgComputeTime = (double)computeTime.count() / (double)iterations;

    double commPerIt = (double)commBytes / (double)iterations;
    statsString += "\nAvg key read time (ms): " + std::to_string(avgKeyReadTime) + "\nAvg compute time (ms): " + std::to_string(avgComputeTime);
    statsString += "\nComm per iteration (B): " + std::to_string(commPerIt);
    stats << statsString;
    stats.close();
    std::cout << statsString << std::endl;
    lossFile.close();
    accFile.close();
    destroyCPURandomness();
    destroyGPURandomness();
}

void evaluatorE2EFakeOffline(std::string modelName, std::string dataset, int party, std::string ip, std::string weightsFile, bool floatWeights, int epochs, int blocks, int blockSz, int batchSz, int H, int W, int C, bool secfloat, bool momentum, std::string keyDir, int device = 0)
{
    WanParameter wanParams;
    AESGlobalContext g;
    initAESContext(&g);    
    initGPURandomness();
    initCPURandomness();
    initGPUMemPool();
    // assert(epochs < 6);

    omp_set_num_threads(2);

    printf("Sync=%d\n", sync);
    printf("Opening fifos\n");
    char one = 1;
    char two = 2;

    auto trainingDir = "output/P" + std::to_string(party) + "/training/";
    auto expName = modelName + "-" + std::to_string(epochs) + "e-" + std::to_string(blocks) + "b";
    auto lossDir = trainingDir + "loss/" + expName + "/";
    auto weightsDir = lossDir + "weights/";
    auto keySzDir = trainingDir + "keysize/";
    std::ofstream lossFile(lossDir + "loss.txt");
    std::ofstream accFile(lossDir + "accuracy.txt");

    dcf::orca::GPUModel<u64> *m = getGPUModel<u64>(modelName, Tensor<u64>(nullptr, {(u64)batchSz, (u64)H, (u64)W, (u64)C}));
    m->setTrain(momentum);
    printf("Model created\n");
    m->initWeights(weightsFile, floatWeights);
    printf("Weights initialized\n");

    u8 *keyBuf1, *keyBuf2, *curKeyBuf, *nextKeyBuf;
    u64 keySz = getKeySz(keySzDir, modelName);
    getAlignedBuf(&keyBuf1, keySz);
    getAlignedBuf(&keyBuf2, keySz);
    int curBuf = 0;
    curKeyBuf = keyBuf1;
    nextKeyBuf = keyBuf2;

    SigmaPeer *peer = new GpuPeer(true);
    LlamaBase<u64> *llama = nullptr;

    // automatically truncates by scale
    LlamaConfig::party = party + 2;
    LlamaConfig::bitlength = dcf::orca::global::bw;
    llama = new LlamaBase<u64>();
    if (LlamaConfig::party == SERVER)
        llama->initServer(ip, (char **)&curKeyBuf);
    else
        llama->initClient(ip, (char **)&curKeyBuf);
    peer->peer = LlamaConfig::peer;

    if (secfloat)
        secfloat_init(party + 1, ip);
    
    std::string keyFile = keyDir + modelName + "_training_key" + std::to_string(party);
    // dropOSPageCache();
    std::chrono::duration<int64_t, std::milli> onlineTime = std::chrono::duration<int64_t, std::milli>::zero();
    std::chrono::duration<int64_t, std::milli> computeTime = std::chrono::duration<int64_t, std::milli>::zero();
    uint64_t keyReadTime = 0;
    size_t commBytes = 0;
    printf("Starting training\n");
    
    Dataset d = readDataset(dataset, party);
    int fd;
    fd = openForReading(keyFile + "_" + to_string(0) + "_" + to_string(0) + "_" + std::to_string(0) + ".dat");
    size_t weightsSize, OptimizerSize;

    u64 *mask_W = NULL;
    string wMaskFile = "";
    wMaskFile = weightsDir + "weights_mask_" + std::to_string(party) + "_" + to_string(0) + "_" + to_string(0) + "_" + std::to_string(0) + ".dat";
    mask_W = (u64 *)readFile(wMaskFile, &weightsSize);
    
    u64 *mask_Opt = NULL;
    string OptMaskFile = "";
    OptMaskFile = weightsDir + "optimizer_mask_" + std::to_string(party) + "_" + to_string(0) + "_" + to_string(0) + "_" + std::to_string(0) + ".dat";
    mask_Opt = (u64 *)readFile(OptMaskFile, &OptimizerSize);
    std::cout << "Opt Mask size="  << OptimizerSize << std::endl;
    for (int l = 0; l < epochs; l++)
    {
        for (int k = 0; k < blocks; k++)
        {            
            peer->sync();
            auto startComm = peer->bytesReceived();
            auto start = std::chrono::high_resolution_clock::now();
            lseek(fd, 0, SEEK_SET);
            for (int j = 0; j < blockSz; j++)
            {
                readKey(fd, keySz, curKeyBuf, &keyReadTime);
                peer->sync();
                auto computeStart = std::chrono::high_resolution_clock::now();
                auto labelsIdx = (k * blockSz + j) * batchSz * d.classes;
                int dataIdx = (k * blockSz + j) * d.H * d.W * d.C * batchSz;
                trainModel(m, &curKeyBuf, party, peer, &(d.data[dataIdx]), &(d.labels[labelsIdx]), &g, secfloat, llama, l, l * blocks * blockSz + k * blockSz + j);
                auto computeEnd = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(computeEnd - computeStart);
                computeTime += elapsed;  
                curKeyBuf = &keyBuf1[0]; 
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            onlineTime += elapsed;
            printf("Online time (ms): %lu\n", elapsed.count());
            auto endComm = peer->bytesReceived();
            commBytes += (endComm - startComm);
            std::pair<double, double> res;
            m->dumpWeights(weightsDir + "masked_weights_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + "_" + std::to_string(blockSz-1) + ".dat");
            m->dumpOptimizer(weightsDir + "masked_optimizer_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + "_" + std::to_string(blockSz-1) + ".dat", party);
            
            string maskedWFile = weightsDir + "masked_weights_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + "_" + std::to_string(blockSz-1) + ".dat";
            std::cout << "===========" << maskedWFile << "==========" << std::endl;
            auto masked_W = (u64 *)readFile(maskedWFile, &weightsSize);
            int N = weightsSize / sizeof(u64);
            auto W = new u64[N];
            
            if (dataset == "mnist")
            {
                std::cout << "Getting loss for MNIST" << std::endl;
                res = getLossMNISTFakeOffline<i64>(modelName, (u64)dcf::orca::global::scale, W, mask_W, masked_W, N);
            }
            else
            {
                std::cout << "Getting loss for CIFAR10" << std::endl;
                res = getLossCIFAR10FakeOffline<i64>(modelName, (u64)dcf::orca::global::scale, W, mask_W, masked_W, N);
            }
            
            for (int i = 0; i < m->layers.size(); i++)
            {
                m->layers[i]->initWeights((u8**)&W, false);
            }
            
            string maskedOptFile = weightsDir + "masked_optimizer_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + "_" + std::to_string(blockSz-1) + ".dat";
            std::cout << "===========" << maskedOptFile << "==========" << std::endl;
            auto masked_Opt = (u64 *)readFile(maskedOptFile, &OptimizerSize);
            
            N = OptimizerSize / sizeof(u64);
            auto Opt = new u64[N];
            std::cout << "Weights file=" << maskedOptFile << ", " << OptMaskFile << std::endl;
            std::cout << "Opt Masked Size=" << N << std::endl;
            for (int i = 0; i < N; i++)
            {
                Opt[i] = masked_Opt[i] - (mask_Opt ? mask_Opt[i] : 0);
                cpuMod(Opt[i], dcf::orca::global::bw);
            }
            for (int i = 0; i < m->layers.size(); i++)
            {
                m->layers[i]->initOptimizer((u8**)&Opt, party);
            }
            assert(std::filesystem::remove(maskedWFile));
            assert(std::filesystem::remove(maskedOptFile));
            auto accuracy = res.first;
            auto loss = res.second;
            printf("Accuracy=%lf, Loss=%lf\n", accuracy, loss);
            lossFile << loss << std::endl;
            accFile << accuracy << std::endl;   
        }
    }
    close(fd);


    LlamaConfig::peer->close();
    int iterations = epochs * blocks * blockSz;
    commBytes += inputOnlineComm;
    commBytes += secFloatComm;
    
    // add the wan_time of softmax
    wan_time += numRounds * wanParams.rtt;
    wan_time += (inputOnlineComm + secFloatComm) / (wanParams.comm_bytes_per_ms);
    
    std::ofstream stats(trainingDir + expName + ".txt");
    auto statsString = "Total time taken (ms): " + std::to_string(onlineTime.count()) + "\nTotal bytes communicated: " + std::to_string(commBytes) + "\nSecfloat softmax bytes: " + std::to_string(inputOnlineComm + secFloatComm);
    statsString += "\nWan extra time taken (ms)" + std::to_string(wan_time);
    auto avgKeyReadTime = (double)keyReadTime / (double)iterations;
    auto avgComputeTime = (double)computeTime.count() / (double)iterations;
    double commPerIt = (double)commBytes / (double)iterations;
    auto wan_extra_time = (double)wan_time / (double)iterations;
    statsString += "\nAvg key read time (ms): " + std::to_string(avgKeyReadTime) + "\nAvg compute time (ms): " + std::to_string(avgComputeTime) + "\nAvg wan extra time (ms)" + std::to_string(wan_extra_time);
    statsString += "\nComm per iteration (B): " + std::to_string(commPerIt);
    stats << statsString;
    stats.close();
    std::cout << statsString << std::endl;
    lossFile.close();
    accFile.close();
    destroyCPURandomness();
    destroyGPURandomness();
}

void evaluatorPerf(std::string modelName, std::string dataset, int party, std::string ip, int iterations, int batchSz, int H, int W, int C, bool secfloat, bool momentum, std::string keyDir)
{
    WanParameter wanParams;
    AESGlobalContext g;
    initAESContext(&g);
    initGPUMemPool();
    initGPURandomness();
    initCPURandomness();

    dcf::orca::GPUModel<u64> *m = getGPUModel<u64>(modelName, Tensor<u64>(nullptr, {(u64)batchSz, (u64)H, (u64)W, (u64)C}));
    m->setTrain(momentum);
    size_t inpMemSz = m->inpSz * sizeof(u64);
    auto inp = (u64 *)cpuMalloc(inpMemSz);
    memset(inp, 0, inpMemSz);
    size_t opMemSz = m->batchSz * m->classes * sizeof(u64);
    auto labels = (u64 *)cpuMalloc(opMemSz);
    memset(labels, 0, opMemSz);

    u8 *keyBuf1, *keyBuf2, *curKeyBuf, *nextKeyBuf;
    auto trainingDir = "output/P" + std::to_string(party) + "/training/";
    auto keySzDir = trainingDir + "keysize/";
    u64 keySz = getKeySz(keySzDir, modelName);
    getAlignedBuf(&keyBuf1, keySz);
    getAlignedBuf(&keyBuf2, keySz);
    int curBuf = 0;
    curKeyBuf = keyBuf1;
    nextKeyBuf = keyBuf2;

    SigmaPeer *peer = new GpuPeer(true);
    LlamaBase<u64> *llama = nullptr;

    LlamaConfig::party = party + 2;
    llama = new LlamaBase<u64>();
    if (LlamaConfig::party == SERVER)
        llama->initServer(ip, (char **)&curKeyBuf);
    else
        llama->initClient(ip, (char **)&curKeyBuf);
    peer->peer = LlamaConfig::peer;

    if (secfloat)
        secfloat_init(party + 1, ip);

    std::string keyFile = keyDir + modelName + "_training_key" + std::to_string(party) + ".dat";
    // dropOSPageCache();
    std::chrono::duration<int64_t, std::milli> onlineTime = std::chrono::duration<int64_t, std::milli>::zero();
    std::chrono::duration<int64_t, std::milli> computeTime = std::chrono::duration<int64_t, std::milli>::zero();
    uint64_t keyReadTime = 0;
    size_t commBytes = 0;
    int fd = openForReading(keyFile);
    auto start = std::chrono::high_resolution_clock::now();
    auto startComm = peer->bytesReceived();
    int float_softmax_time = 0;
    for (int j = 0; j < iterations; j++)
    {
        readKey(fd, keySz, curKeyBuf, &keyReadTime);   
        auto computeStart = std::chrono::high_resolution_clock::now();
        trainModelPerf(m, &curKeyBuf, party, peer, inp, labels, &g, secfloat, llama, 0, 0, float_softmax_time);
        auto computeEnd = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(computeEnd - computeStart);
        computeTime += elapsed;
        curKeyBuf = &keyBuf1[0];
    }
    auto end = std::chrono::high_resolution_clock::now();
    onlineTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Online time (ms): %lu\n", onlineTime.count());
    auto endComm = peer->bytesReceived();
    commBytes += (endComm - startComm);
    close(fd);

    // add the online comm of softmax
    commBytes += inputOnlineComm;
    commBytes += secFloatComm;
    
    std::cout << "numRounds: " << numRounds << std::endl;
    std::cout << "inputOnlineComm: " << inputOnlineComm << std::endl;
    // add the wan_time of softmax
    wan_time += numRounds * wanParams.rtt;
    wan_time += (inputOnlineComm + secFloatComm) / (wanParams.comm_bytes_per_ms);
    LlamaConfig::peer->close();
    std::ofstream stats(trainingDir + modelName + ".txt");
    auto statsString = "\n" + modelName + "\n";
    statsString += "Total time taken (ms): " + std::to_string(onlineTime.count()) + "\nTotal bytes communicated: " + std::to_string(commBytes) + "\nSecfloat softmax bytes: " + std::to_string(inputOnlineComm + secFloatComm);
    statsString += "\nWan extra time taken (ms)" + std::to_string(wan_time);
    statsString += "\nFLoat SoftMax time take (ms): " + std::to_string(float_softmax_time) + "\n";
    statsString += "\nIterations: " + std::to_string(iterations) + "\n";
    auto totTimeByIt = (double)onlineTime.count() / (double)(iterations - 1);
    auto avgKeyReadTime = (double)keyReadTime / (double)iterations;
    auto avgComputeTIme = (double)computeTime.count() / (double)iterations;
    auto wan_extra_time = (double)wan_time / (double)iterations;
    int truncateComm = 0;
    for (int i = 0; i < m->layers.size(); i++)
    {
        truncateComm += m->layers[i]->s.truncate_comm_bytes;
    }
    double commPerIt = (double)commBytes / (double)iterations;
    statsString += "\nTotal time / iterations (ms): " + std::to_string(totTimeByIt) + "\nAvg key read time (ms): " + std::to_string(avgKeyReadTime) + "\nAvg compute time (ms): " + std::to_string(avgComputeTIme) + "\nAvg wan extra time (ms): " + std::to_string(wan_extra_time);
    statsString += "\nComm per iteration (B): " + std::to_string(commPerIt) + "\n";  
    statsString += "\nTruncate Comm (B): " + std::to_string(truncateComm / 8 * (double)(iterations)) + "\nAvg Truncate Comm (B): " + std::to_string((double)truncateComm / 8);
    stats << statsString;
    stats.close();
    std::cout << statsString << std::endl;
    std::cout << float_softmax_time << std::endl;
    destroyCPURandomness();
    destroyGPURandomness();
}

int global_device = 0;
double wan_time = 0;
int main(int argc, char *argv[])
{
    sytorch_init();
    auto keyDir = std::string(argv[1]);
    auto experiment = std::string(argv[2]);
    auto ip = std::string(argv[3]);
    int party = atoi(argv[4]);
    global_device = atoi(argv[5]);
    using T = u64;
    // Neha: need to fix this later 

    if (experiment.compare("CNN2-FLOAT") == 0)
    {
        int epochs = 2;
        int blocks = 46;
        int blockSz = 10; // 600
        int batchSz = 128;
        evaluatorE2EFakeOffline("CNN2", "mnist", party, ip, "weights/CNN2.dat", false, epochs, blocks, blockSz, batchSz, 28, 28, 1, true, true, keyDir);
    }
    else if (experiment.compare("CNN3-FLOAT") == 0){
        int epochs = 2;
        int blocks = 78;
        int blockSz = 10;
        int batchSz = 64;
        evaluatorE2EFakeOffline("CNN3", "cifar10", party, ip, "weights/CNN3.dat", false, epochs, blocks, blockSz, batchSz, 32, 32, 3, true, true, keyDir);
    }
    else if (experiment.compare("P-SecureML-FLOAT") == 0)
    {
        int epochs = 2;
        int blocks = 46;
        int blockSz = 10;
        int batchSz = 128;
        evaluatorE2EFakeOffline("P-SecureML", "mnist", party, ip, "weights/PSecureMlNoRelu.dat", false, epochs, blocks, blockSz, batchSz, 28, 28, 1, true, true, keyDir);   
    }
    if (experiment.compare("CNN2-COMM") == 0){
        int iterations = 11;
        int batchSz = 128;
        evaluatorPerf("CNN2", "mnist", party, ip, iterations, batchSz, 28, 28, 1, true, true, keyDir);
    }
    else if (experiment.compare("CNN3-COMM") == 0){
        int iterations = 11;
        int batchSz = 64;
        evaluatorPerf("CNN3", "cifar10", party, ip, iterations, batchSz, 32, 32, 3, true, true, keyDir);
    }
    else if (experiment.compare("P-VGG16") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        evaluatorPerf("P-VGG16", "cifar10", party, ip, iterations, batchSz, 32, 32, 3, true, true, keyDir);
    }
    else if (experiment.compare("P-AlexNet") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        evaluatorPerf("P-AlexNet", "cifar10", party, ip, iterations, batchSz, 32, 32, 3, true, true, keyDir);
    }
    else if (experiment.compare("P-LeNet") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        evaluatorPerf("P-LeNet", "mnist", party, ip, iterations, batchSz, 28, 28, 1, true, true, keyDir);
    }
    else if (experiment.compare("P-SecureML") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        evaluatorPerf("P-SecureML", "mnist", party, ip, iterations, batchSz, 28, 28, 1, true, true, keyDir);
    }
    else if (experiment.compare("AlexNet") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        evaluatorPerf("AlexNet", "cifar10", party, ip, iterations, batchSz, 32, 32, 3, true, true, keyDir);
    }
    else if (experiment.compare("Pattern1") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        // 这里设置float softmax是为了更好的减去通信开销，计算pattern
        evaluatorPerf("Pattern1", "mnist", party, ip, iterations, batchSz, 28, 28, 1, true, true, keyDir);
    }
    else if (experiment.compare("Pattern2") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        // 这里设置float softmax是为了更好的减去通信开销，计算pattern
        evaluatorPerf("Pattern2", "mnist", party, ip, iterations, batchSz, 28, 28, 1, true, true, keyDir);
    }
    return 0;
}
