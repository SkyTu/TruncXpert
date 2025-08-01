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

#pragma once

#include "utils/gpu_comms.h"
#include "fss/wing/gpu_relu.h"
#include "gpu_layer.h"
#include "fss/gpu_select.h"


namespace wing
{

    template <typename T>
    class ReluExtendLayer : public GPULayer<T>
    {
    public:
        int bin, bout, /*f,*/ numRelus;
        GPUReluZeroExtKey<T> reluExtendKey;
        // GPUReluExtendKey<T> reluExtendKey;
        u32 *drelu;
        u8 *dReluMask;
        GPUSelectExtKey<T> backpropSelectExtKey;
        GPUSelectKey<T> backpropSelectKey;
        bool nextBackExt;
        // AESGlobalContext* gaes;
        // Stats s;

        ReluExtendLayer(int bin, int bout, int numRelus, bool nextBackExt);
        ~ReluExtendLayer();
        T *genForwardKey(uint8_t **key_as_bytes, int party, T *d_inputMask, AESGlobalContext *gaes);
        T *genBackwardKey(uint8_t **key_as_bytes, int party, T *d_incomingGradMask, AESGlobalContext *gaes, int epoch);
        void readForwardKey(uint8_t **key_as_bytes);
        void readBackwardKey(uint8_t **key_as_bytes, int epoch);
        T *forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *g);
        T *backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *g, int epoch);
    };
}


#include "relu_extend_layer.cu"