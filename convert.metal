//
//  convert.metal
//  effort
//
//  Created 02/04/2024.
//

#include <metal_stdlib>
using namespace metal;


//kernel void getProbes(const device bfloat *w[[buffer(0)]],

kernel void getProbes(const device half *w[[buffer(0)]],
                      device half *probes[[buffer(1)]],
                      const device uint& cols [[buffer(2)]],
                      const device uint& probeRowRepeat [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
    for (uint i = 0; i<probeRowRepeat; i++) {
        probes[id*probeRowRepeat+i] = half(w[id+i + id*cols]);
    }
}
/*
 gpu.deploy("prepareValsIdxs", buffers: [w, wVals, wIds], ints:[w.rows, w.cols], threadCount: w.rows)

 */
kernel void prepareValsIdxs(const device half *w[[buffer(0)]],

//kernel void prepareValsIdxs(const device bfloat *w[[buffer(0)]],
                            device half *vals[[buffer(1)]],
                            device ushort *idxs[[buffer(2)]],
                            const device uint& srcRows[[buffer(3)]],
                            const device uint& srcCols[[buffer(4)]],
                            uint rowId [[thread_position_in_grid]]) {
    
    for (uint i = 0; i<srcCols; i++) {
        vals[i*srcRows + rowId] = half(w[rowId*srcCols + i]);
        idxs[i*srcRows + rowId] = rowId;
    }
    
}

kernel void preBucketize(const device half *_wVals [[buffer(0)]],
                      const device ushort *_wIdxs [[buffer(1)]],
                      device half* _bVals [[buffer(2)]],
                      const device uint& inDim [[buffer(3)]],
                      const device uint& outDim [[buffer(4)]],
                      const device uint& bSize [[buffer(5)]],
                      uint rowId [[thread_position_in_grid]]) {
    
    const uint __offset = rowId*(outDim/bSize)*bSize;
    const device half *wVals = &_wVals[__offset];
    const device ushort *wIdxs = &_wIdxs[__offset];
    
    const uint _offset = rowId*(outDim/bSize)*(bSize + 1);
    device half *bVals = &_bVals[_offset];

    for (uint i=0; i<outDim; i++) {
        half val = wVals[i];
        ushort idx = wIdxs[i];
        const ushort bucket = idx / bSize;
        const ushort posId = idx % bSize;
        
        // encode position in the least significant bits
        // a bit of noise never hurt nobody ;)
        ushort valCast = as_type<ushort>(val);
        ushort mask = 0xFFFF ^ (bSize-1);
        valCast &= mask;
        valCast |= posId;
        val = as_type<half>(valCast);
        
        // bvals is [counter, val1, val2...], [counter, val1, val2...]
        ushort bOffset = bucket * (bSize+1);
        ushort counter = bVals[bOffset];
        bVals[bOffset+1+counter] = val;
        bVals[bOffset] += 1;
    }
}
/*
let buckets = Matrix(shape:[inDim*bSize, outDim/bSize])
gpu.deploy("bucketize", buffers: [bVals, buckets], ints:[inDim, outDim, bSize], threadCount: [inDim, bSize])
*/
kernel void bucketize(const device half *bVals [[buffer(0)]],
                      device half *buckets [[buffer(1)]],
                      const device uint& inDim [[buffer(2)]],
                      const device uint& outDim [[buffer(3)]],
                      const device uint& bSize [[buffer(4)]],
                      uint2 id [[thread_position_in_grid]]) {
    uint bucketCount = outDim/bSize;
    uint rowId = id.x;
    uint bucketNo = id.y;

    for (uint i = 0; i<bSize; i++) {
        uint bValsRowSize = (bSize+1)*(outDim/bSize);
        half val = bVals[(1+i)+bucketNo*(bSize+1)+(rowId*bValsRowSize)];
        
        uint bucketsRowOffset = rowId*bucketCount + i*inDim*bucketCount;
        buckets[bucketNo + bucketsRowOffset] = val;
    }
}
/*
 gpu.deploy("makeStats", buffers: [buckets, stats], ints:[buckets.cols], threadCount: buckets.rows)

 */
kernel void makeStats(const device half *buckets [[buffer(0)]],
                      device half4 *stats [[buffer(1)]],
                      const device uint& bCols [[buffer(2)]],
                      uint rowId [[thread_position_in_grid]]) {

    float sum = 0;
    for (uint i=0; i<bCols; i++) {
        sum += abs(buckets[rowId*bCols+i]);
    }
    stats[rowId].x = sum / bCols;
    stats[rowId].y = sum / bCols;
    stats[rowId].z = sum / bCols;
    stats[rowId].w = sum / bCols;

}

/*
gpu.deploy("extract", buffers: [slice, newSlice, outliers],
                      ints: [newSlice.cols],
                      float16s: [minCutoff, maxCutoff, minRange, maxRange],
                      threadCount: slice.rows)
 */
kernel void cojest3() {
    
    for (int i = 0; i<10000; i++) {
        //mielimy
    }
}

kernel void cojest() {
    
    for (int i = 0; i<10000; i++) {
        //mielimy
    }
}

kernel void cojest2() {
    
    for (int i = 0; i<10000; i++) {
        //mielimy
    }
}


kernel void extract(device half *slice[[buffer(0)]],
                    device uchar* newSlice[[buffer(1)]],
                    device half2* outliers[[buffer(2)]],
                    device const int& sliceCols[[buffer(3)]],
                    device const float& minCutoff[[buffer(4)]],
                    device const float& maxCutoff[[buffer(5)]],
                    device const float& minRange[[buffer(6)]],
                    device const float& maxRange[[buffer(7)]],
                    uint rowNum [[thread_position_in_grid]]
                    ){
    device half* sliceRow = &slice[rowNum*sliceCols];
    device uchar* out = &newSlice[rowNum*sliceCols];
    /*
    const ushort maxOutliersSize = 32;
    ushort outliersSize = 0;
    device half2* outliersRow = &outliers[rowNum*32];
    */
    
    // remove outliers. first priority to high outliers
    /*
    for (int i = 0; i<sliceCols && outliersSize < maxOutliersSize; i++) {
        if (abs(sliceRow[i]) > maxCutoff) {
            outliersRow[outliersSize].x = sliceRow[i];
            outliersRow[outliersSize].y = i;
            sliceRow[i] = 0;
            outliersSize += 1;
        }
    }
    
    for (int i = 0; i<sliceCols && outliersSize < maxOutliersSize; i++) {
        if (abs(sliceRow[i]) < minCutoff) {
            outliersRow[outliersSize].x = sliceRow[i];
            outliersRow[outliersSize].y = i;
            sliceRow[i] = 0;
            outliersSize += 1;
        }
    }
    */
    // effortize!
    for (int i = 0; i<sliceCols; i++) {
        float el = abs(sliceRow[i]);
        if (el == 0) {continue;}

        uchar elSign = sliceRow[i]>0?0:0x80;
        
        uchar bit4;
        float step = (maxRange-minRange)/16;
        
        el -= minRange;
        el /= step;
        el = max(0.0, el);
        el = min(el, 15.0);
        bit4 = el;
        bit4 <<= 3;
        
        uchar position = as_type<ushort>(sliceRow[i]) & 7;
        
        out[i] = elSign | bit4 | position;
    }
    
}




kernel void findPercentile(device const half *probes [[buffer(0)]],
                           device float* out[[buffer(1)]],
                           constant uint &_perc [[buffer(2)]],

                           uint id [[thread_position_in_grid]],
                           uint tiisg [[thread_index_in_simdgroup]],
                           uint siitg [[simdgroup_index_in_threadgroup]],
                           uint tpg [[threads_per_grid]]) {

    // damn this func be ugly
    
    uint effort = 4096-_perc;
    half myMax = -999;
    half myMin = 999;
    half4 myVal;
    
    for (int i = 0; i<4; i++) {
        myVal[i] = abs(probes[4*id]);
        myMax = max(myMax, myVal[i]);
        myMin = min(myMin, myVal[i]);
    }
    
    //half myVal = abs(v[id]*probes[id+expNo[0]*4096]);
    half sgMin = simd_min(myMin);
    half sgMax = simd_max(myMax);
    
    threadgroup half tgMin[32] = {999};
    threadgroup half tgMax[32] = {-999};
    
    threadgroup float minBound = 999;
    threadgroup float maxBound = -999;
    threadgroup float newBound = -999;
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
            out[0] = newBound;
            return;
        }
    }
}


kernel void idxsBitonicSortAbs(device half     *floats     [[ buffer(0) ]],
                             device ushort     *idxs     [[ buffer(1) ]],
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
    bool isHigherIndexGreaterThanLower = (abs(floats[gid]) < abs(floats[gidPlusDistance]));
    if (isGidDistanceZero && isHigherIndexGreaterThanLower == direction) {
        float temp = floats[gid];
        floats[gid] = floats[gidPlusDistance];
        floats[gidPlusDistance] = temp;

        temp = idxs[gid];
        idxs[gid] = idxs[gidPlusDistance];
        idxs[gidPlusDistance] = temp;
    }
}
