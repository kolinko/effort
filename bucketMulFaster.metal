//
//  bucketMulFaster.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 12/04/2024.
//

#include <metal_stdlib>
using namespace metal;


#define CUTOFF_SCALE 100000
// ^ cheap hack to prevent cutoff from going out of range in wv wq in Mistral
// although a better idea would be to rescale all the weights in models by ~1000 to better fit
// the half precision.
//
// btw. the code should use bfloats instead of halfs for weights, but then it would be a bit more
// hussle to implement

kernel void prepareExpertDispatchFaster(device const float* v[[buffer(0)]],
                                  device const half4* binStats[[buffer(1)]],
                                  device const int* expertNo[[buffer(2)]],
                                  device const float* cutoff[[buffer(3)]],
                                  device float2* dispatch[[buffer(4)]],
                                  device atomic_uint* dispatchCount[[buffer(5)]],
                                  device const int& chunkSize [[buffer(6)]],
                                  device const uint& rowsCount [[buffer(7)]],
                                  device const uint& colsCount [[buffer(8)]],
                                  device const int& expertSize[[buffer(9)]],
                                  uint id [[thread_position_in_grid]]) {
    uint dispatchOffset = expertSize * expertNo[0];
    uint begin = chunkSize * id + dispatchOffset;
    uint end = begin + chunkSize;
    
    int idx;
    const uint idxIncr = 1;
    ushort counter = idxIncr;
    
    for (uint i = begin; i<end; i++) {
        half4 s = binStats[i]; // row, min, max, mean
        float val = v[i % rowsCount]; // int(s[0])
        if (cutoff[0] < CUTOFF_SCALE * float(s[3]) * abs(val)) {
            if (counter == idxIncr) {
                idx = atomic_fetch_add_explicit(dispatchCount, idxIncr, memory_order_relaxed);
                counter = 0;
            }
            dispatch[idx+counter] = {val, float(i*colsCount)};
            counter += 1;
        }
    }
    
}

# define STEP 4

kernel void bucketMulFaster(
                   device const half *weights [[buffer(0)]],
                   device const float2 *dispatch [[buffer(1)]],
                   device float *result [[buffer(2)]],
                   constant uint *dispatchSize [[buffer(3)]],
                   constant uint &cols [[buffer(4)]],
                   constant uint &groups [[buffer(5)]],
                   uint2 id [[thread_position_in_grid]]) {
                      
    float myVal[16] = {0};
      
    const uint rowOffset = id.y*dispatchSize[0]/groups;
    for (uint r=0; r<dispatchSize[0]/groups; r+=STEP) {
//        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int s=0; s<STEP; s++) { // for better optimisation
            float2 d = dispatch[rowOffset + r + s];//+s
            half w = weights[int(d[1]) + id.x];
            float v = d[0]*float(w);
            ushort pos = as_type<ushort>(w)&15;
            
            for (int i=0; i<16; i++) {
                myVal[i] += (pos == i) ? v : 0;
            }
            
        }
    }

    uint myOff = (id.y*16384);
    for (int i = 0; i<16; i++) {
        result[myOff + id.x*16 + i] = myVal[i];
      //  atomic_fetch_add_explicit(&result[id.x*16+i], myVal[i], memory_order_relaxed);
    }
                          
}

