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

/*for i in 0..<probes {
    o[i] = abs(v[i] * weights[i*weights.cols! + i])
}*/

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

#define SIZE 4096/4


kernel void bucketMulRows(device const half *vector [[buffer(0)]],
                   device const half4 *weights [[buffer(1)]],
                   device float *result [[buffer(2)]],
                   device const half *seedVec [[buffer(3)]],
                   constant int &inDim [[buffer(4)]],
                   constant int &outDim [[buffer(5)]],

                  uint id [[thread_position_in_grid]]) {
    
    //
    // for groups == 1: go through every row of inDim, fetch its outCol value and += to myVal
    // for groups > 1 : go through rows groupId*groupSize to (groupId+1)*groupSize, fetch its outCol value and += to myVal
    
    int groups = 64;
    int inDim4 = inDim/4;
    int rowsPerGroup = inDim4 / groups;
    int outCol = id / groups;
    int outGroup = id % groups;
    int rowBegin = outGroup * rowsPerGroup;
    int rowEnd = (outGroup+1) * rowsPerGroup;
    float myVal = 0;
    half4 w1;
    half4 w2;
    half4 w3;
    half4 w4;
//    int counter = 0;
    for (int row = rowBegin; row < rowEnd; row+=4) {
        w1 = weights[row];
        w2 = weights[row+1];
        w3 = weights[row+2];
        w4 = weights[row+3];
        myVal += w1.x + w1.y + w1.z + w1.w;
        myVal += w2.x + w2.y + w2.z + w2.w;
        myVal += w3.x + w3.y + w3.z + w3.w;
        myVal += w4.x + w4.y + w4.z + w4.w;
    }
    result[outCol] += myVal;
//    atomic_fetch_add_explicit(&result[outCol], myVal, memory_order_relaxed);
    
}


kernel void bucketMul(device const half *vector [[buffer(0)]],
                   device const float4 *weights [[buffer(1)]],
                   device float *result [[buffer(2)]],
                   device const half *seedVec [[buffer(3)]],
                   constant int &inDim [[buffer(4)]],
                   constant int &outDim [[buffer(5)]],

                  uint id [[thread_position_in_grid]]) {
    
    //
    // for groups == 1: go through every row of inDim, fetch its outCol value and += to myVal
    // for groups > 1 : go through rows groupId*groupSize to (groupId+1)*groupSize, fetch its outCol value and += to myVal
    
    int groups = 32;
    int inDim4 = inDim/4;
    int rowsPerGroup = inDim4 / groups;
    int outCol = id / groups;
    int outGroup = id % groups;
    int rowBegin = outGroup * rowsPerGroup;
    int rowEnd = (outGroup+1) * rowsPerGroup;
    float myVal[16*4];
    int row = rowBegin;
    float4 w1;
    for (int i = 0; i<250; i+=1) { //
        w1 = weights[row+i]; //*(int(vector[i])&7)
        myVal[int(w1.x)&15] += vector[i] * (w1.x);
        myVal[int(w1.y)&15+16] += vector[i] * (w1.y);
        myVal[int(w1.z)&15+32] += vector[i] * (w1.z);
        myVal[int(w1.w)&15+46] += vector[i] * (w1.w);
    }
    for(int i = 0; i<4; i++) {
        int off = i*16;
        result[off+outCol] += myVal[off+0];
        result[off+outCol+1] += myVal[off+1];
        result[off+outCol+2] += myVal[off+2];
        result[off+outCol+3] += myVal[off+3];
        result[off+outCol+4] += myVal[off+4];
        result[off+outCol+5] += myVal[off+5];
        result[off+outCol+6] += myVal[off+6];
        result[off+outCol+7] += myVal[off+7];
        result[off+outCol+8] += myVal[off+8];
        result[off+outCol+9] += myVal[off+9];
        result[off+outCol+10] += myVal[off+10];
        result[off+outCol+11] += myVal[off+11];
        result[off+outCol+12] += myVal[off+12];
        result[off+outCol+13] += myVal[off+13];
        result[off+outCol+14] += myVal[off+14];
        result[off+outCol+15] += myVal[off+15];
    }

//    atomic_fetch_add_explicit(&result[outCol], myVal, memory_order_relaxed);
    
}
/*
kernel void bucketMul(device const half *vector [[buffer(0)]],
                   device const half4 *weights [[buffer(1)]],
                   device float *result [[buffer(2)]],
                   constant int &inDim [[buffer(3)]],
                   constant int &outDim [[buffer(4)]],
                   uint id [[thread_position_in_grid]]) {
    
    //
    // for groups == 1: go through every row of inDim, fetch its outCol value and += to myVal
    // for groups > 1 : go through rows groupId*groupSize to (groupId+1)*groupSize, fetch its outCol value and += to myVal
    
    int groups = 64;
    int inDim4 = inDim/4;
    int rowsPerGroup = inDim4 / groups;
    int outCol = id / groups;
    int outGroup = id % groups;
    int rowBegin = outGroup * rowsPerGroup;
    int rowEnd = (outGroup+1) * rowsPerGroup;
    float myVal = 0;
    half4 w1;
    half4 w2;
    half4 w3;
    half4 w4;
//    int counter = 0;
    for (int row = rowBegin; row < rowEnd; row+=4) { //4+4*(int(myVal)&1)
        w1 = weights[row];
        w2 = weights[row+1];
        w3 = weights[row+2];
        w4 = weights[row+3];
        myVal += w1.x + w1.y + w1.z + w1.w;
        myVal += w2.x + w2.y + w2.z + w2.w;
        myVal += w3.x + w3.y + w3.z + w3.w;
        myVal += w4.x + w4.y + w4.z + w4.w;
    }
    result[outCol] += myVal;
    
    /*
    int groups = 64;
    int inDim4 = inDim/4;
    int rowsPerGroup = inDim4 / groups;
    int outCol = id / groups;
    int outGroup = id % groups;
    int rowBegin = outGroup * rowsPerGroup;
    int rowEnd = (outGroup+1) * rowsPerGroup;
    float myVal = 0;
    for (int row = rowBegin; row < rowEnd; row+=2) {
        half4 w1 = weights[row];
        half4 w2 = weights[row+1];
        myVal += w1.x + w1.y + w1.z + w1.w;
        myVal += w2.x + w2.y + w2.z + w2.w;
    }
    atomic_fetch_add_explicit(&result[outCol], myVal, memory_order_relaxed);
     
    
}*/
    

kernel void accum(device const half *vector [[buffer(0)]],
                  device half2 *weights [[buffer(1)]],
//                  device const ushort4 *rowIds [[buffer(1)]],
//                  device const half4 *rowVals [[buffer(2)]],
                  device float *counter [[buffer(3)]],
                  constant half *cutoff [[buffer(4)]],
                  constant int &innerDim [[buffer(5)]],
                  constant int &outerDim [[buffer(6)]],
                  uint id [[thread_position_in_grid]],
                  uint tpitg [[thread_position_in_threadgroup]],
                  uint sid [[thread_index_in_simdgroup]]) {
    
    
#define yolo 1
    
#ifdef yolo
    /*
    half myVal = vector[id];
    int offset = id*outerDim/4;
    half myEndVal = 0;
    threadgroup half4 tgRowVal;
    threadgroup ushort4 tgRowId;
    for (int i = 0; i < innerDim; i++) {
        for (int j = 0; j < SIZE/8; j++) {
            if (tpitg == 0) {
                    tgRowVal = rowVals[i*SIZE + j];
                    tgRowId = rowIds[i*SIZE + j];
            };
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int x = 0; x<4; x++) {
                 myEndVal += (id == tgRowId[x]) * tgRowVal[x];
            };
        };
    }
    */
    /*
    int outDim = 11008;
    int inDim = 4096;
    int bucketSize = 16;
    int bucketCount = outDim / bucketSize;
    int myBucket = id;
    float buckets[16];
    
    for (int i = 0; i<inDim; i++) {
        half2 h = weights[i*bucketCount + myBucket];
        buckets[int(h.x) % 16] += h.y;
    }
    
    for (int i = 0; i<16; i++) {
        result[myBucket*bucketSize + i] = buckets[i]
    }*/
    
    /*
    half myVal = vector[id];
    int offset = id*outerDim/4;
    half myEndVal = 0;
    threadgroup half4 tgRowVals[SIZE/8];
    threadgroup ushort4 tgRowIds[SIZE/8];
    for (int i = 0; i < innerDim; i++) {
        int off = tpitg % (SIZE / 8);
        tgRowVals[off] = rowVals[i*SIZE + off];
        tgRowIds[off] = rowIds[i*SIZE + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < SIZE/8; j++) {
            for (int x = 0; x<4; x++) {
                myEndVal += tgRowVals[j][x] * (id == tgRowIds[j][x]);
            };
        };
    }*/
    
   // counter[id] = myEndVal;

    
    
    /*
    for (int i = 0; i < 687; i+=2) { // outerDim/4
//        simdgroup_half8x8 hlf;
        simd_half4 hlf;
        hlf = rowVals[i];
        myEndVal += hlf[sid % 4];
        
//        hlf = rowVals[i];
        
        half4 out = rowVals[offset+i]*myVal;
        half4 out2 = rowVals[offset+i+1]*myVal;

        ushort4 rid = rowIds[offset+i];
        ushort4 rid2 = rowIds[offset+i+1];
         */
        /*
        counter[rid[0]] += out[0];
        counter[rid[1]] += out[1];
        counter[rid[2]] += out[2];
        counter[rid[3]] += out[3];
        counter[rid2[0]] += out2[0];
        counter[rid2[1]] += out2[1];
        counter[rid2[2]] += out2[2];
        counter[rid2[3]] += out2[3];
         */
        /*
        atomic_fetch_add_explicit(&counter[rid[0]], out[0], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid[1]], out[1], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid[2]], out[2], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid[3]], out[3], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid2[0]], out2[0], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid2[1]], out2[1], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid2[2]], out2[2], memory_order_relaxed);
        atomic_fetch_add_explicit(&counter[rid2[3]], out2[3], memory_order_relaxed);
         */
        
       /*
      if (abs(out[3]) < abs(cutoff[0])) {
            break;
        }
        
    }*/
//    atomic_fetch_add_explicit(&counter[id], myEndVal, memory_order_relaxed);
    
#else
    half myVal = vector[id];
    int offset = id*outerDim/4;
    
    for (int i = 0; i < 2000; i+=2) { // outerDim/4
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
        
       /*
      if (abs(out[3]) < abs(cutoff[0])) {
            break;
        }
        */
    }
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
//        sum += matrix[11008*i+id] * vector[i];

    }

    result[row] = sum;
}


kernel void mul_col_11008(device const half *matrix [[buffer(0)]],
                    device const half *vector [[buffer(1)]],
                    device half *result [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    half sum = 0.0;
    int row = id;
    int offset = id * 11008;
    
    for (int i = 0; i < 11008; i++) {
        sum += matrix[(offset+i)] * vector[i];
    }

    result[row] = sum;
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


