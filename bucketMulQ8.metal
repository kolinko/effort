//
//  bucketMulQ8.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 08/04/2024.
//

#include <metal_stdlib>
using namespace metal;


//
//  mulQ8.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 03/04/2024.
//

#include <metal_stdlib>
using namespace metal;


/*
 
 gpu.deploy("bucketOutliersQ8",
            buffers: [outliers, dispatch, out, dispatch.size],
            ints: [groups],
            threadCount: [maxOutliers, groups],
            threadGroupSize: [32, 1, 1])
 
 */

kernel void decodeQ8(
                     device const uchar *v [[buffer(0)]],
                     device half2 *out [[buffer(1)]],
                     device const float &minRange [[buffer(2)]],
                     device const float &step [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
                         
                         ushort w = v[id];
                         // decode sign
                         const half sign = w & 0x80 ? -1 : 1;
                         
                         // get position
                         ushort j = (w & 7)+id*8;
                         
                         w >>= 3;
                         w &= 15;
                         // calc proper value
                         // half val = 0.015 * float(w);
                         
                         half val = step * float(w);
                         val += minRange;
                         val *= sign;
                         
                         out[id] = val;
                         out[id].y = as_type<half>(j);
                         // as_type<half>(ushort(as_type<ushort>(val) | w>0x80?0x8:0));
/*
                         for (int i=0; i<8; i++) {
                             myVal[i+y*8] += (j == i)?float(w)*float(d[0]):0;
                         }
                         
                         w = as_type<float>(as_type<ushort>(w) >> 8);*/
                         
                         
                     }

kernel void bucketOutliersQ8(
                        device const half2 *outliers [[buffer(0)]],
                        device const float4 *dispatch [[buffer(1)]],
                        device atomic_float *result [[buffer(2)]],
                        constant float *dispatchSize [[buffer(3)]],
                        constant int &groups [[buffer(5)]],
                        uint3 id [[thread_position_in_grid]]) {
    
//    float myVal[16] = {0};
    const uint rowOffset = id.y*dispatchSize[0]/groups;
    for (int r=0; r<dispatchSize[0]/groups; r+=8) {
        for (int s=0; s<8; s++) { // for better optimisation
            float4 d = dispatch[rowOffset + r+s];
            half2 o = outliers[uint(d[1])];
            float val = float(d[0]) * o[0];

            atomic_fetch_add_explicit(&result[uint(o[1])], val, memory_order_relaxed);

        }
    }
                      
                          
}


kernel void bucketMulQ8v3(
                        device const ushort *weights [[buffer(0)]],
                        device const float4 *dispatch [[buffer(1)]],
                        device atomic_float *result [[buffer(2)]],
                        constant float *dispatchSize [[buffer(3)]],
                        constant uint &cols [[buffer(4)]],
                        constant int &groups [[buffer(5)]],
                        uint3 id [[thread_position_in_grid]]) {
    
    float myVal[16] = {0};
    const uint dsize = dispatchSize[0];
    const uint rowOffset = id.y*dsize/groups;
    const uint end = dsize/groups;
    const uint step = 1;
    for (uint r=0; r<end; r+=step) {
        for (ushort s=0; s<step; s++) { // for better optimisation
            float4 d = dispatch[rowOffset + r+s];
            ushort w = weights[as_type<uint>(d[1]) + id.x];

            for (ushort y=0; y<2; y++) {
                const half sign = w & 0x80 ? -1 : 1;
                
                // get position
                ushort j = w & 7;
                
                w >>= 3;
                float val = d[3] * float(w&15);
                val += d[2];
                val *= sign;
                
                for (int i=0; i<8; i++) {
                    myVal[i+y*8] += (j == i)?float(val)*float(d[0]):0; // little vs small endian
                }
                
                w >>= 5;
                
            }

        }
    }
                      
    for (int i = 0; i<16; i++) {
//        result[id.x*16+i] += myVal[i];//, memory_order_relaxed);
        atomic_fetch_add_explicit(&result[id.x*16+i], myVal[i], memory_order_relaxed);
    }
                          
}


kernel void bucketMulQ8(
                        device const ushort *weights [[buffer(0)]],
                        device const float4 *dispatch [[buffer(1)]],
                        device atomic_float *result [[buffer(2)]],
                        constant float *dispatchSize [[buffer(3)]],
                        constant uint &cols [[buffer(4)]],
                        constant int &groups [[buffer(5)]],
                        uint3 id [[thread_position_in_grid]]) {
    
    float myVal[16] = {0};
    const uint dsize = dispatchSize[0];///2;
    const uint rowOffset = id.y*dsize/groups;
    const uint end = dsize/groups;
    const uint step = 1;
    for (uint r=0; r<end; r+=step) {
        for (ushort s=0; s<step; s++) { // for better optimisation
            float4 d = dispatch[rowOffset + r+s];
            ushort w = weights[as_type<uint>(d[1]) + id.x];

            for (ushort y=0; y<2; y++) {
                const half sign = w & 0x80 ? -1 : 1;
                
                // get position
                ushort j = w & 7;
                
                w >>= 3;
                float val = d[3] * float(w&15);
                val += d[2];
                val *= sign;
                
                for (int i=0; i<8; i++) {
                    myVal[i+y*8] += (j == i)?float(val)*float(d[0]):0; // little vs small endian
                }
                
                w >>= 5;
                
            }

        }
    }
                      
    for (int i = 0; i<16; i++) {
        atomic_fetch_add_explicit(&result[id.x*16+i], myVal[i], memory_order_relaxed);
    }
                          
}

/*
 
 gpu.deploy("bucketIntegrate", buffers: [tmpMulVec, out],
            threadCount: [32, out.rows, 1],
            threadGroupSize: [32, 1, 1])

 
 */

kernel void bucketMulQ8v4(
                        device const ushort *weights [[buffer(0)]],
                        device const float4 *dispatch [[buffer(1)]],
                        device float *result [[buffer(2)]],
                        constant float *dispatchSize [[buffer(3)]],
                        constant uint &cols [[buffer(4)]],
                        constant int &groups [[buffer(5)]],
                        uint3 id [[thread_position_in_grid]]) {
    
    float myVal[16] = {0};
    const uint dsize = dispatchSize[0];
    const uint rowOffset = id.y*dsize/groups;
    const uint end = dsize/groups;
    const uint step = 1;
    for (uint r=0; r<end; r+=step) {
        for (ushort s=0; s<step; s++) { // for better optimisation
            float4 d = dispatch[rowOffset + r+s];
            ushort w = weights[as_type<uint>(d[1]) + id.x];

            for (ushort y=0; y<2; y++) {
                const half sign = w & 0x80 ? -1 : 1;
                
                // get position
                ushort j = w & 7;
                
                w >>= 3;
                float val = d[3] * float(w&15);
                val += d[2];
                val *= sign;
                
                for (int i=0; i<8; i++) {
                    myVal[i+y*8] += (j == i)?float(val)*float(d[0]):0; // little vs small endian
                }
                
                w >>= 5;
                
            }

        }
    }
            
    uint myOff = (id.y*16384);
    for (int i = 0; i<16; i++) {
        result[myOff+id.x*16+i] = myVal[i];
//        atomic_fetch_add_explicit(&result[id.x*16+i], myVal[i], memory_order_relaxed);
    }
                          
}
/*
#define tmpMulVecMaxSize = 16384
kernel void bucketIntegrate(device const float* tmpMulVec[[buffer(0)]],
                            device float* out[[buffer(1)]],
                            uint2 id [[thread_position_in_grid]],
                            uint tiisg [[thread_index_in_simdgroup]]
                            ) {
    
    float sum = tmpMulVec[id.y+(tiisg*2*16384)] + tmpMulVec[id.y+(tiisg*2+1)*16384];
    sum = simd_sum(sum);
    if (tiisg == 0) {
        out[id.y] = sum;
    }
    
}*/


// same as regular, but appends slice encoding.
kernel void prepareExpertDispatchQ8(device const float* v[[buffer(0)]],
                                  device const half4* binStats[[buffer(1)]],
                                  device const int* expertNo[[buffer(2)]],
                                  device const half* cutoff[[buffer(3)]],
                                  device float4* dispatch[[buffer(4)]],
                                  device atomic_float* dispatchCount[[buffer(5)]],
                                  device const float2* stats[[buffer(6)]],
                                  device const int& chunkSize [[buffer(7)]],
                                  device const uint& rowsCount [[buffer(8)]],
                                  device const uint& colsCount [[buffer(9)]],
                                  device const int& expertSize[[buffer(10)]],
                                  uint id [[thread_position_in_grid]]) {
    uint dispatchOffset = expertSize * expertNo[0];
    uint begin = chunkSize * id + dispatchOffset;
    uint end = begin + chunkSize;
    
    int idx;
    const uint idxIncr = 1;
    ushort counter = idxIncr;
    
    for (uint i = begin; i<end; i++) {
        half4 s = binStats[i];
        float val = v[i % rowsCount]; // int(s[0])
        float ucomp = float(s.z) * abs(val);
        if (true) {//cutoff[0] < ucomp) {
            if (counter == idxIncr) {
                idx = atomic_fetch_add_explicit(dispatchCount, idxIncr, memory_order_relaxed);
                counter = 0;
            }

            uint pos = float(i)*float(colsCount);
            ushort numSlice = (i-dispatchOffset) / rowsCount;

            dispatch[idx+counter] = {val, as_type<float>(pos), stats[numSlice].x, stats[numSlice].y};
            counter += 1;
        }
    }
    
}
