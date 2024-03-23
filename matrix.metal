//
//  matrix.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 26/01/2024.
//

#include <metal_stdlib>
using namespace metal;

kernel void silu(device const float *x1 [[buffer(0)]],
                 device const float *x3 [[buffer(1)]],
                 device half *out [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    out[id] = x3[id] * x1[id] / (1 + exp(-x1[id]));
}


kernel void probe(device const half *v [[buffer(0)]],
                  device const half *weights [[buffer(1)]],
                  device half *out[[buffer(2)]],
                  constant int &wCols [[buffer(3)]],
                  uint id [[thread_position_in_grid]]) {
    out[id] = abs(v[id] * weights[id*wCols + id]);
}

kernel void getVal(device const half* vector [[buffer(0)]],
                   device half *val [[buffer(1)]],
                   const device uint &pos [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {
    if (id == pos) {
        val[0] = vector[pos];
    }
}

/*
 
 bucketMul

 
 */

kernel void prepareDispatch(device const half* v[[buffer(0)]],
                            device const half4* binStats[[buffer(1)]],
                            device const half* cutoff[[buffer(2)]],
                            device float2* dispatch[[buffer(3)]],
                            device atomic_float* dispatchCount[[buffer(4)]],
                            device const int& chunkSize [[buffer(5)]],
                            uint id [[thread_position_in_grid]]){
    
    for (uint i = chunkSize*id; i<(id+1)*chunkSize; i++) {
        half4 s = binStats[i]; // row, min, max, mean
        float val = v[int(s[0])];
        if (cutoff[0] < float(s[3]) * abs(val)) {
            int idx = atomic_fetch_add_explicit(dispatchCount, 1, memory_order_relaxed);
            dispatch[idx][0] = val;
            dispatch[idx][1] = i;
        }
    }
    
}

kernel void bucketMul(
                   device const half *weights [[buffer(0)]],
                   device const float2 *dispatch [[buffer(1)]],
                   device atomic_float *result [[buffer(2)]],
                   constant float *dispatchSize [[buffer(3)]],
                   constant uint &cols [[buffer(4)]],
                   constant int &groups [[buffer(5)]],
                   ushort2 id [[thread_position_in_grid]]) {
                      
    float myVal[16] = {0};
      
    const ushort rowOffset = id.y*dispatchSize[0]/groups;
    for (int r=0; r<dispatchSize[0]/groups; r+=32) {
        for (int s=0; s<32; s++) { // for better optimisation
            
            float2 d = dispatch[rowOffset + r+s];
            half w = weights[int(d[1])*cols + id.x];
            for (int i=0; i<16; i++) {
                myVal[i] += ((as_type<ushort>(w)&15) == i)?d[0]*float(w):0;
            }
            
        }
    }
                      
    for (int i = 0; i<16; i++) {
      atomic_fetch_add_explicit(result+(id.x*16+i), myVal[i], memory_order_relaxed);
    }
                          
}



#define outer_count 4096

kernel void internal(device const half *fxn [[buffer(0)]],
                     device const half *w1 [[buffer(1)]],
                     device const half *w3 [[buffer(2)]],
                     device half *result [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    half x1 = 0.0;
    half x2 = 0.0;
    half x3 = 0.0;
    int offset = id*outer_count;

    for (int i = 0; i<outer_count; i++) {
        half x = fxn[i];
        x1 += x * w1[offset + i];
        x3 += x * w3[offset + i];
    }
    x2 = x3 * x1 / (1 + exp(-x1));
    result[id] = x2;
}

kernel void second (device const half *matrix [[buffer(0)]],
                    device const half *vector [[buffer(1)]],
                    device half *result [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    half sum = 0.0;
    int offset = id * 11008;
    for (int i = 0; i < 11008; i++) {
        sum += matrix[(offset + i)] * vector[i];
    }

    result[id] += sum;
}

kernel void basicBitonicSort(device half     *floats     [[ buffer(0) ]],
                             constant int     &p             [[ buffer(1) ]],
                             constant int     &q             [[ buffer(2) ]],
                             uint             gid         [[ thread_position_in_grid ]])
{
    // taken from https://developer.apple.com/forums/thread/674181
    //            https://github.com/tgymnich/MetalSort

    int pMinusQ = p-q;
    int distance = 1 << pMinusQ;
    uint gidShiftedByP = gid >> p;
    // True: Increasing / False: Descreasing
    bool direction = (gidShiftedByP & 2) == 0;
    uint gidDistance = (gid & distance);
    bool isGidDistanceZero = (gidDistance == 0);
    uint gidPlusDistance = (gid | distance);
    bool isLowerIndexGreaterThanHigher = (floats[gid] > floats[gidPlusDistance]);
    if (isGidDistanceZero && isLowerIndexGreaterThanHigher == direction) {
        float temp = floats[gid];
        floats[gid] = floats[gidPlusDistance];
        floats[gidPlusDistance] = temp;
    }
}
