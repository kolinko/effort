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

kernel void probeExpert(device const float *v [[buffer(0)]],
                        device const half *probes [[buffer(1)]],
                        device const uint *expNo [[buffer(2)]],
                        device half *out[[buffer(3)]],
                        constant int &wCols [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = abs(v[id] * probes[id+expNo[0]*4096]);
}

//gpu.deploy("findCutoff", buffers: [v, ew.probes, expNo, cutoff2], ints:[q], threadCount: probesCount, threadGroupSize: [1024, 1, 1])


kernel void findCutoff(device const float *v [[buffer(0)]],
                       device const half *probes [[buffer(1)]],
                       device const uint *expNo [[buffer(2)]],
                       device half* out[[buffer(3)]],

                       constant uint &_quant [[buffer(4)]],

                       uint id [[thread_position_in_grid]],
                       uint tiisg [[thread_index_in_simdgroup]],
                       uint siitg [[simdgroup_index_in_threadgroup]],
                       uint tpg [[threads_per_grid]]) {

//    const uint probesCount = 4096;
    uint quant = 4096-_quant;
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
            
            if (countAbove < quant) {
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

        if ((globalCount == quant) ||
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

/*
 
 bucketMul

 */


kernel void prepareExpertDispatch(device const float* v[[buffer(0)]],
                                  device const half4* binStats[[buffer(1)]],
                                  device const int* expertNo[[buffer(2)]],
                                  device const half* cutoff[[buffer(3)]],
                                  device float2* dispatch[[buffer(4)]],
                                  device atomic_float* dispatchCount[[buffer(5)]],
                                  device const int& chunkSize [[buffer(6)]],
                                  device const uint& rowsCount [[buffer(7)]],
                                  device const int& expertSize[[buffer(8)]],
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
