//
//  wild.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 09/04/2024.
//

#include <metal_stdlib>
using namespace metal;




/*

kernel void bucketMulWild(
                        device const half *weights [[buffer(0)]],
                        device const float2 *dispatch [[buffer(1)]],
                        device float *result [[buffer(2)]],
                        constant float *dispatchSize [[buffer(3)]],
                        constant uint &cols [[buffer(4)]],
                        constant int &groups [[buffer(5)]],
                        uint id [[thread_position_in_grid]],
                        uint tpitg [[thread_position_in_threadgroup]],
                        uint tgpig [[threadgroup_position_in_grid]],
                        uint tiisg [[thread_index_in_simdgroup]],
                        uint sgiitg [[simdgroup_index_in_threadgroup]]) {
                            
    if (true) { // stupid autoformatter
        float2 myCounter = 0;
        const ushort myPos = tiisg % 16;
        const ushort myOff = sgiitg*2;

        for(uint row = 0; row < uint(dispatchSize[0]); row++){
            threadgroup half w[64];
            threadgroup float2 d;
            simdgroup_half8x8 tmp_w;
            if (sgiitg==0) {
                d = dispatch[row];
                simdgroup_load(tmp_w, &weights[uint(d[0])]);
                simdgroup_store(tmp_w, w);
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            const half2 myW = {w[myOff], w[myOff+1]};
            const bool isMine = (as_type<ushort>(myW.x) & 15) == myPos;
            myCounter.x += isMine ? myW.x : 0;
            const bool isMine2 = (as_type<ushort>(myW.y) & 15) == myPos;
            myCounter.y += isMine ? myW.y : 0;
        }
        
    }
  }

*/
