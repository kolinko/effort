//
//  bucketMulQ4.metal
//  effort
//
//  Created by Tomasz Kolinko on 23/04/2024.
//

#include <metal_stdlib>
using namespace metal;


// effort given to optimise this: 0%
kernel void calcOutliers(device const float* v[[buffer(0)]],
                         device const float4* outliers[[buffer(1)]],
                         device atomic_float* out[[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
        
    float4 o = outliers[id];
    atomic_fetch_add_explicit(&out[uint(o.z)], v[uint(o.y)] * o.x, memory_order_relaxed);
    
}

#define CUTOFF_SCALE 100000

kernel void prepareDispatchQ4(device const float* v[[buffer(0)]],
                                  device const float2* binStats[[buffer(1)]],
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
        float2 s = binStats[i]; // avg abs x 2
        float val = v[i / 8];// % rowsCount - accidental different organisation than regular bucketMul - to be fixed
        if (cutoff[0] < CUTOFF_SCALE * float(s[1]) * abs(val)) {
            if (counter == idxIncr) {
                idx = atomic_fetch_add_explicit(dispatchCount, idxIncr, memory_order_relaxed);
                counter = 0;
            }
            dispatch[idx+counter] = {val*s[1], float(i*colsCount)}; // multiplying by avg here
            counter += 1;
        }
    }
    
}

# define STEP 4

kernel void bucketMulQ4(
                   device const ushort *weights [[buffer(0)]],
                   device const float2 *dispatch [[buffer(1)]],
                   device atomic_float *result [[buffer(2)]],
                   constant uint *dispatchSize [[buffer(3)]],
                   constant uint &groups [[buffer(4)]],
                   uint2 id [[thread_position_in_grid]]) {
                      
    float myVal[8*4] = {0}; // switch to bfloat to save mem/registries?
      
    const uint rowOffset = id.y*dispatchSize[0]/groups;
    for (uint r=0; r<dispatchSize[0]/groups; r+=STEP) {
        for (int s=0; s<STEP; s++) { // gets unrolled and speeds up calcs
            float2 d = dispatch[rowOffset + r + s];
            ushort w = weights[int(d[1]) + id.x];
            for (int i = 3; i>=0; i--) {
                float v = w & 8 ? -d[0] : d[0];
                ushort pos = w&7;
                myVal[pos+i*8] += v;
                w >>= 4;
            }
        }
    }

//    uint myOff = (id.y*16384);
    for (int i = 0; i<8*4; i++) {
//        result[myOff + id.x*16 + i] = myVal[i]; // todo: get back to this. atomic_fetch is
                                                  // used for faster dev & testing, but slows down, of course
        atomic_fetch_add_explicit(&result[id.x*8*4+i], myVal[i], memory_order_relaxed);
    }
                          
}


#define INTSTEPS  4
#define tmpMulVecMaxSize = 16384
kernel void bucketIntegrateQ4(device const float* tmpMulVec[[buffer(0)]],
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
