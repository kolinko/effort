//
//  bucketMul.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 09/04/2024.
//

#include <metal_stdlib>
using namespace metal;



kernel void prepareExpertDispatchFast(device const float* v[[buffer(0)]],
                                  device const half4* binStats[[buffer(1)]],
                                  device const int* expertNo[[buffer(2)]],
                                  device const half* cutoff[[buffer(3)]],
                                  device float2* dispatch[[buffer(4)]],
                                  device atomic_float* dispatchCount[[buffer(5)]],
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
        if (cutoff[0] < float(s[3]) * abs(val)) {
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

kernel void bucketMulFast(
                   device const half *weights [[buffer(0)]],
                   device const float2 *dispatch [[buffer(1)]],
                   device atomic_float *result [[buffer(2)]],
                   constant float *dispatchSize [[buffer(3)]],
                   constant uint &cols [[buffer(4)]],
                   constant int &groups [[buffer(5)]],
                   uint2 id [[thread_position_in_grid]]) {
                      
    float myVal[16] = {0};
      
    const uint rowOffset = id.y*dispatchSize[0]/groups;
    for (int r=0; r<dispatchSize[0]/groups; r+=STEP) {
        for (int s=0; s<STEP; s++) { // for better optimisation
            
            float2 d = dispatch[rowOffset + r];//+s
            half w = weights[int(d[1]) + id.x];
            for (int i=0; i<16; i++) {
                myVal[i] += ((as_type<ushort>(w)&15) == i)?d[0]*float(w):0;
            }
        }
    }
                      
    for (int i = 0; i<16; i++) {
      atomic_fetch_add_explicit(&result[id.x*16+i], myVal[i], memory_order_relaxed);
    }
                          
}
