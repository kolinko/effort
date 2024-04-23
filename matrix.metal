//
//  matrix.metal
//  effort
//
//  Created 26/01/2024.
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

kernel void silu32b(device const float *x1 [[buffer(0)]],
                 device const float *x3 [[buffer(1)]],
                 device float *out [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    uint begin = id*64;
    uint end = id*64+64;
    for (uint i=begin; i<end; i++) {
        out[i] = x3[i] * x1[i] / (1 + exp(-x1[i]));
    }
}



kernel void findCutoff(device const float *v [[buffer(0)]],
                       device const half *probes [[buffer(1)]],
                       device const uint *expNo [[buffer(2)]],
                       device half* out[[buffer(3)]],

                       constant uint &_effort [[buffer(4)]],

                       uint id [[thread_position_in_grid]],
                       uint tiisg [[thread_index_in_simdgroup]],
                       uint siitg [[simdgroup_index_in_threadgroup]],
                       uint tpg [[threads_per_grid]]) {

//    const uint probesCount = 4096;
    uint effort = 4096-_effort;
    half myMax = -999;
    half myMin = 999;
    half4 myVal;
    
    for (int i = 0; i<4; i++) {
        myVal[i] = abs(v[4*id+i]*probes[4*id+i+expNo[0]*4096]);
        myMax = max(myMax, myVal[i]);
        myMin = min(myMin, myVal[i]);
    }
    
    //half myVal = abs(v[id]*probes[id+expNo[0]*4096]);
    half sgMin = simd_min(myMin);
    half sgMax = simd_max(myMax);
    
    threadgroup half tgMin[32] = {999};
    threadgroup half tgMax[32] = {-999};
    
    threadgroup half minBound = 999;
    threadgroup half maxBound = -999;
    threadgroup half newBound = -999;
    threadgroup short minCount = 0;
    threadgroup short maxCount = 0;
    
    if (tiisg == 0) {
        tgMin[siitg] = sgMin;
        tgMax[siitg] = sgMax;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (siitg == 0) {
        sgMin = tgMin[tiisg];
        sgMax = tgMax[tiisg];
        
        minBound = simd_min(sgMin);
        maxBound = simd_max(sgMax);
    }

    
    threadgroup short tgAbove[32] = {0};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    newBound = (minBound + maxBound)/2;

    ushort loops = 0;
    minCount = 4096;
    while (true) {
        loops += 1;
        ushort countAbove = 0;
        ushort myAbove = 0;
        threadgroup ushort globalCount = 0;
        for (int i = 0; i<4; i++) {
            myAbove += myVal[i] > newBound ? 1 : 0;
        }

        tgAbove[siitg] = simd_sum(myAbove);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (siitg == 0) {
            countAbove = tgAbove[tiisg];
            countAbove = simd_sum(countAbove);
            
            if (countAbove < effort) {
                maxBound = newBound;
                maxCount = countAbove;
            } else {
                minBound = newBound;
                minCount = countAbove;
            }
            
            newBound = (maxBound+minBound)/2;
            globalCount = countAbove;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((globalCount == effort) ||
            (maxBound - minBound < 0.00001) ||
            (abs(maxCount - minCount) < 2)) {
            if (id == 0){
                out[0] = newBound;
            }
            return;
        }
        
        if (loops>100) {
            return;
        }
    }
}


kernel void getVal(device const half* vector [[buffer(0)]],
                   device half *val [[buffer(1)]],
                   const device uint &pos [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {
    if (id == pos) {
        val[0] = vector[pos];
    }
}


kernel void basicMul(device const float* v[[buffer(0)]],
                     device const half* m[[buffer(1)]],
                     device float* out[[buffer(2)]],
                     const device uint &numCols[[buffer(3)]],
                     uint rowId [[thread_position_in_grid]]) {
    
    float sum = 0;
    uint offset = rowId*numCols;
    for (uint i=0; i<numCols; i++) {
        sum += v[i]*m[i+offset];
    }
    out[rowId] = sum;
}


kernel void basicMul2(device const float* v[[buffer(0)]],
                     device const half* m[[buffer(1)]],
                     device atomic_float* out[[buffer(2)]],
                     const device uint &numCols[[buffer(3)]],
                     const device uint &chunkSize[[buffer(4)]],
                     uint2 rowId [[thread_position_in_grid]]) {
    
    float sum = 0;
    uint offset = rowId.x*numCols + rowId.y*chunkSize;
    for (uint i=0; i<chunkSize; i++) {
        sum += v[rowId.y*chunkSize+i]*m[i+offset];
    }
    atomic_fetch_add_explicit(out+rowId.x, sum, memory_order_relaxed);
//    out[rowId.x] = sum;
}


/*
 
 bucketMul

 */



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
