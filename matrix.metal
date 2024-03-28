//
//  matrix.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 26/01/2024.
//

#include <metal_stdlib>
using namespace metal;

kernel void silu(device const half *x1 [[buffer(0)]],
                 device const half *x3 [[buffer(1)]],
                 device half *out [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    out[id] = float(x3[id]) * float(x1[id]) / (1 + exp(-float(x1[id])));
}

kernel void silu32(device const float *x1 [[buffer(0)]],
                 device const float *x3 [[buffer(1)]],
                 device float *out [[buffer(2)]],
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

kernel void probeShort(device const half *v [[buffer(0)]],
                  device const half *probes [[buffer(1)]],
                  device half *out[[buffer(2)]],
                  constant int &wCols [[buffer(3)]],
                  uint id [[thread_position_in_grid]]) {
    out[id] = abs(v[id] * probes[id]);
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


kernel void prepareDispatch32(device const float* v[[buffer(0)]],
                            device const half4* binStats[[buffer(1)]],
                            device const half* cutoff[[buffer(2)]],
                            device float2* dispatch[[buffer(3)]],
                            device atomic_float* dispatchCount[[buffer(4)]],
                            device const int& chunkSize [[buffer(5)]],
                            device const uint& rowsCount [[buffer(6)]],
                            uint id [[thread_position_in_grid]]){

    int idx;
    const uint idxIncr = 1;
    ushort counter = idxIncr;
    
    for (uint i = chunkSize*id; i<(id+1)*chunkSize; i++) {
        half4 s = binStats[i]; // row, min, max, mean
        float val = v[i % rowsCount]; // int(s[0])
        if (cutoff[0] < float(s[3]) * abs(val)) {
            if (counter == idxIncr) {
                idx = atomic_fetch_add_explicit(dispatchCount, idxIncr, memory_order_relaxed);
                counter = 0;
            }
            dispatch[idx+counter] = {val, float(i)};
            counter += 1;
        }
    }
    
}

kernel void prepareDispatch(device const half* v[[buffer(0)]],
                            device const half4* binStats[[buffer(1)]],
                            device const half* cutoff[[buffer(2)]],
                            device float2* dispatch[[buffer(3)]],
                            device atomic_float* dispatchCount[[buffer(4)]],
                            device const int& chunkSize [[buffer(5)]],
                            device const uint& rowsCount [[buffer(6)]],
                            uint id [[thread_position_in_grid]]){

    int idx;
    const uint idxIncr = 1;
    ushort counter = idxIncr;
    
    for (uint i = chunkSize*id; i<(id+1)*chunkSize; i++) {
        half4 s = binStats[i]; // row, min, max, mean
        float val = v[i % rowsCount]; // int(s[0])
        if (cutoff[0] < float(s[3]) * abs(val)) {
            if (counter == idxIncr) {
                idx = atomic_fetch_add_explicit(dispatchCount, idxIncr, memory_order_relaxed);
                counter = 0;
            }
            dispatch[idx+counter] = {val, float(i)};
            counter += 1;
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
                   uint2 id [[thread_position_in_grid]]) {
                      
    float myVal[16] = {0};
      
    const uint rowOffset = id.y*dispatchSize[0]/groups;
    for (int r=0; r<dispatchSize[0]/groups; r+=1) { //32) {
    //    for (int s=0; s<32; s++) { // for better optimisation
            
            float2 d = dispatch[rowOffset + r];//+s
            half w = weights[int(d[1])*cols + id.x];
            for (int i=0; i<16; i++) {
                myVal[i] += ((as_type<ushort>(w)&15) == i)?d[0]*float(w):0;
            }
//        }
    }
                      
    for (int i = 0; i<16; i++) {
      atomic_fetch_add_explicit(result+(id.x*16+i), myVal[i], memory_order_relaxed);
    }
                          
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
