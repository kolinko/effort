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
                   device float *result [[buffer(2)]],
                   constant float *dispatchSize [[buffer(3)]],
                   constant uint &cols [[buffer(4)]],
                   constant int &groups [[buffer(5)]],
                   uint2 id [[thread_position_in_grid]]) {
                      
    float myVal[16] = {0};
      
    const uint rowOffset = id.y*dispatchSize[0]/groups;
    for (int r=0; r<dispatchSize[0]/groups; r+=STEP) {
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


#define INTSTEPS  4
#define tmpMulVecMaxSize = 16384
kernel void bucketIntegrate(device const float* tmpMulVec[[buffer(0)]],
                            device float* out[[buffer(1)]],
                            uint2 id [[thread_position_in_grid]],
                            uint tiisg [[thread_index_in_simdgroup]]
                            ) {
    uint begin = INTSTEPS * id.y;
    uint end = begin + INTSTEPS;
    for (uint i=begin; i<end; i++){
        float sum = tmpMulVec[i+(tiisg*16384)];// + tmpMulVec[id.y+(tiisg*2+1)*16384];
        sum = simd_sum(sum);
        if (tiisg == 0) {
            out[i] = sum;
        }
    }
        
}

/*

kernel void bucketMulWild(
                        device const half *weights [[buffer(0)]],
                        device const float2 *dispatch [[buffer(1)]],
                        device float *result [[buffer(2)]],
                        constant float *dispatchSize [[buffer(3)]],
                        constant uint &cols [[buffer(4)]],
                        constant int &groups [[buffer(5)]],
                        uint id [[thread_position_in_grid]],
                        uint tpitg [[thread_position_in_threadgroup]],
                        uint tgpig [[threadgroup_position_in_grid]],
                        uint tiisg [[thread_index_in_simdgroup]],
                        uint sgiitg [[simdgroup_index_in_threadgroup]]) {
                            
    if (true) { // stupid autoformatter
        float2 myCounter = 0;
        const ushort myPos = tiisg % 16;
        const ushort myOff = sgiitg*2;

        for(uint row = 0; row < uint(dispatchSize[0]); row++){
            threadgroup half w[64];
            threadgroup float2 d;
            simdgroup_half8x8 tmp_w;
            if (sgiitg==0) {
                d = dispatch[row];
                simdgroup_load(tmp_w, &weights[uint(d[0])]);
                simdgroup_store(tmp_w, w);
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            const half2 myW = {w[myOff], w[myOff+1]};
            const bool isMine = (as_type<ushort>(myW.x) & 15) == myPos;
            myCounter.x += isMine ? myW.x : 0;
            const bool isMine2 = (as_type<ushort>(myW.y) & 15) == myPos;
            myCounter.y += isMine ? myW.y : 0;
        }
        
    }
  }

*/
