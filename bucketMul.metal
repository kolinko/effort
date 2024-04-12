//
//  bucketMul.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 09/04/2024.
//

#include <metal_stdlib>
using namespace metal;
//gpu.deploy("zeroRange32", buffers: [bm.dispatch, prevSize, bm.dispatch.size], threadCount: 2024 )

kernel void zeroRange32(device float2* dispatch [[buffer(0)]],
                        device uint* begin [[buffer(1)]],
                        device uint* end [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    
    uint myPos = *begin + id;
    if (myPos < *end) {
        dispatch[myPos] = {0, 0};
    }
}

kernel void roundUp(device uint* result [[buffer(0)]],
                  device uint* prevResult [[buffer(1)]],
                  device const uint& number [[buffer(2)]],

                     uint id [[thread_position_in_grid]]) {
    
    prevResult[0] = result[0];
    result[0] = (1+uint(uint(result[0])/number)) * number;

}

#define CUTOFF_SCALE 100000
// ^ cheap hack to prevent cutoff from going out of range in wv wq in Mistral
// although a better idea would be to rescale all the weights in models by ~1000 to better fit
// the half precision.
//
// btw. the code should use bfloats instead of halfs for weights, but then it would be a bit more
// hussle to implement 

kernel void prepareExpertDispatchFast(device const float* v[[buffer(0)]],
                                  device const half4* binStats[[buffer(1)]],
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
        half4 s = binStats[i]; // row, min, max, mean
        float val = v[i % rowsCount]; // int(s[0])
        if (cutoff[0] < CUTOFF_SCALE * float(s[3]) * abs(val)) {
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
                   constant uint *dispatchSize [[buffer(3)]],
                   constant uint &cols [[buffer(4)]],
                   constant uint &groups [[buffer(5)]],
                   uint2 id [[thread_position_in_grid]]) {
                      
    float myVal[16] = {0};
      
    const uint rowOffset = id.y*dispatchSize[0]/groups;
    for (uint r=0; r<dispatchSize[0]/groups; r+=STEP) {
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



kernel void findCutoff32(device const float *v [[buffer(0)]],
                       device const half *probes [[buffer(1)]],
                       device const uint *expNo [[buffer(2)]],
                       device float* out[[buffer(3)]],

                       constant uint &_quant [[buffer(4)]],

                       uint id [[thread_position_in_grid]],
                       uint tiisg [[thread_index_in_simdgroup]],
                       uint siitg [[simdgroup_index_in_threadgroup]],
                       uint tpg [[threads_per_grid]]) {

//    const uint probesCount = 4096;
    uint quant = 4096-_quant;
    float myMax = -999;
    float myMin = 999;
    bfloat4 myVal;
    
    for (int i = 0; i<4; i++) {
        myVal[i] = bfloat(CUTOFF_SCALE * abs(v[4*id+i]*bfloat(probes[4*id+i+expNo[0]*4096])));
        myMax = max(myMax, myVal[i]);
        myMin = min(myMin, myVal[i]);
    }
    
    //half myVal = abs(v[id]*probes[id+expNo[0]*4096]);
    float sgMin = simd_min(myMin);
    float sgMax = simd_max(myMax);
    
    threadgroup bfloat tgMin[32] = {bfloat(999.0)};
    threadgroup bfloat tgMax[32] = {bfloat(-999)};
    
    threadgroup float minBound = 999;
    threadgroup float maxBound = -999;
    threadgroup float newBound = -999;
    threadgroup short minCount = 0;
    threadgroup short maxCount = 0;
    
    if (tiisg == 0) {
        tgMin[siitg] = bfloat(sgMin);
        tgMax[siitg] = bfloat(sgMax);
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
            (maxBound - minBound < 0.00001*CUTOFF_SCALE) ||
            (abs(maxCount - minCount) < 3)) {
            if (id == 0){
                out[0] = newBound;
            }
            return;
        }
        
        if (loops>100) {
            // inf loop? sth went terribly wrong here. it should throw assert during debug
            // and in other case - either newBound and hope for the best, or max range to slow down
            // but guarantee proper results
            if (id == 0) {
                out[0] = newBound;
            }
            return;

        }
    }
}
