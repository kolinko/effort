//
//  matrix.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 26/01/2024.
//

#include <metal_stdlib>
using namespace metal;

kernel void accum(device const half *vector [[buffer(0)]],
                  device const ushort4 *rowIds [[buffer(1)]],
                  device const half4 *rowVals [[buffer(2)]],
                  constant half &cutoff [[buffer(4)]],
                  device atomic_float *counter [[buffer(3)]],
                  uint id [[thread_position_in_grid]]) {
    
    
#define yolo 0
    
#ifdef yolo
    half myVal = vector[id];
    int offset = id*11008/4;
    
    for (int i = 0; i < 11008/4; i+=2) {//1008; i++) {
        half4 out = rowVals[offset+i]*myVal;
        half4 out2 = rowVals[offset+i+1]*myVal;

        ushort4 rid = rowIds[offset+i];
        ushort4 rid2 = rowIds[offset+i+1];

        atomic_fetch_add_explicit(&counter[rid[0]], out[0], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid[1]], out[1], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid[2]], out[2], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid[3]], out[3], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid2[0]], out2[0], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid2[1]], out2[1], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid2[2]], out2[2], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid2[3]], out2[3], memory_order_relaxed);

        
        
      if (abs(out[3]) < abs(cutoff)) {
            break;
        }
    }
    
#else
    half sum = 0.0;
    int row = id;
    int offset = id * 4096;

    for (int i = 0; i < 11008; i++) {
        sum += rowVals[(offset+i)] * vector[i];
    }

    result[row] = sum;
#endif
}



kernel void mul_col_4096(device const half *matrix [[buffer(0)]],
                    device const half *vector [[buffer(1)]],
                    device half *result [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    half sum = 0.0;
    int row = id;
    int offset = id * 4096;
    for (int i = 0; i < 4096; i++) {
        sum += matrix[(offset+i)] * vector[i];
    }

    result[row] = sum;
}

#define outer_count 4096

//constant int outer_count = 4096;

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

kernel void bitonicSort(device float     *floats     [[ buffer(0) ]],
                        device int        *uInts         [[ buffer(1) ]],
                        constant int     &p             [[ buffer(2) ]],
                        constant int     &q             [[ buffer(3) ]],
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
        int temp2 = uInts[gid];
        uInts[gid] = uInts[gidPlusDistance];
        uInts[gidPlusDistance] = temp2;
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


