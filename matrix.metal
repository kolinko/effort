//
//  matrix.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 26/01/2024.
//

#include <metal_stdlib>
using namespace metal;

kernel void accum(device const half *vector [[buffer(0)]],
                  device const ushort *rowIds [[buffer(1)]],
                  device const half *rowVals [[buffer(2)]],
                  device half *result [[buffer(5)]],
                  constant half &cutoff [[buffer(4)]],
                  device atomic_float *counter [[buffer(3)]],
                  uint id [[thread_position_in_grid]]) {
    
    half myVal = vector[id];
    int offset = id*11008;
    
    for (int i = 0; i < 11008; i+=8) {//1008; i++) {
        
        half out = myVal*rowVals[offset+i];
        half out2 = myVal*rowVals[offset+i+1];
        half out3 = myVal*rowVals[offset+i+2];
        half out4 = myVal*rowVals[offset+i+3];
        half out5 = myVal*rowVals[offset+i+4];
        half out6 = myVal*rowVals[offset+i+5];
        half out7 = myVal*rowVals[offset+i+6];
        half out8 = myVal*rowVals[offset+i+7];
        
        short rid = rowIds[offset+i];
        short rid2 = rowIds[offset+i+1];
        short rid3 = rowIds[offset+i+2];
        short rid4 = rowIds[offset+i+3];
        short rid5 = rowIds[offset+i+4];
        short rid6 = rowIds[offset+i+5];
        short rid7 = rowIds[offset+i+6];
        short rid8 = rowIds[offset+i+7];
        /*half out9 = myVal*rowVals[offset+i+8];
        half out10 = myVal*rowVals[offset+i+9];
        half out11 = myVal*rowVals[offset+i+10];
        half out12 = myVal*rowVals[offset+i+11];
        half out13 = myVal*rowVals[offset+i+12];
        half out14 = myVal*rowVals[offset+i+13];
        half out15 = myVal*rowVals[offset+i+14];
        half out16 = myVal*rowVals[offset+i+15];*/
        
        atomic_fetch_add_explicit(&counter[rid], out, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid2], out2, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid3], out3, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid4], out4, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid5], out5, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid6], out6, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid7], out7, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid8], out8, memory_order_relaxed);
        /*atomic_fetch_add_explicit(&counter[rowIds[offset+i+8]], out9, memory_order_relaxed);
        
        atomic_fetch_add_explicit(&counter[rowIds[offset+i+9]], out10, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rowIds[offset+i+10]], out11, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rowIds[offset+i+11]], out12, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rowIds[offset+i+12]], out13, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rowIds[offset+i+13]], out14, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rowIds[offset+i+14]], out15, memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rowIds[offset+i+15]], out16, memory_order_relaxed);*/
        /*
        short rid = rowIds[offset+i];
        short rid2 = rowIds[offset+i+1];
        short rid3 = rowIds[offset+i+2];
        short rid4 = rowIds[offset+i+3];
        short rid5 = rowIds[offset+i+4];
        short rid6 = rowIds[offset+i+5];
        short rid7 = rowIds[offset+i+6];
        short rid8 = rowIds[offset+i+7];
        short rid9 = rowIds[offset+i+8];
        short rid10 = rowIds[offset+i+9];
        short rid11 = rowIds[offset+i+10];
        short rid12 = rowIds[offset+i+11];
        short rid13 = rowIds[offset+i+12];
        short rid14 = rowIds[offset+i+13];
        short rid15 = rowIds[offset+i+14];
        short rid16 = rowIds[offset+i+15];

        result[id+1] += out;
        result[id+2] += out2;
        result[id+3] += out3;
        result[id+4] += out4;
        result[id+5] += out5;
        result[id+6] += out6;
        result[id+7] += out7;
        result[id+8] += out8;
        result[id+9] += out9;
        result[id+10] += out10;
        result[id+11] += out11;
        result[id+12] += out12;
        result[id+13] += out13;
        result[id+14] += out14;
        result[id+15] += out15;
        result[id+16] += out16;
         */
        //        uint rid = rowIds[offset+i];
//        result[rid] += out;
      if (abs(out) < abs(cutoff)) {
            break;
        }
    }
    
    /*
    half sum = 0.0;
    int row = id;
    int offset = id * 4096;

    for (int i = 0; i < 11008; i++) {
        sum += rowVals[(offset+i)] * vector[i];
    }

    result[row] = sum;
     */
    
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


