//
//  wild.metal
//  effort
//
//  Created 09/04/2024.
//

#include <metal_stdlib>
using namespace metal;





kernel void bucketMulWild( device const half *weights [[buffer(0)]],
                        device const float2 *dispatch [[buffer(1)]],
                        device float *result [[buffer(2)]],
                        constant uint *dispatchSize [[buffer(3)]],
                        constant uint &cols [[buffer(4)]],
                        constant int &mulGroups [[buffer(5)]],
                        uint2 id [[thread_position_in_grid]],
                        uint2 tpitg [[thread_position_in_threadgroup]],
                        uint2 tgpig [[threadgroup_position_in_grid]],
                        uint tiisg [[thread_index_in_simdgroup]],
                          uint sgiitg [[simdgroup_index_in_threadgroup]]) {
    const ushort myOff = (sgiitg * 2) + (tiisg & 16);
    const ushort myJ = tiisg & 15;
    const uint chunk = *dispatchSize / mulGroups;
    const uint begin = chunk * id.y;
    const uint end = chunk + begin;
    const uint wOff = tgpig.x * 64;
    // dispatch[0] -> val; dispatch[1] -> address
    float myVal = 0;
    
    for (uint i=begin; i<end; i++) {
        float2 d;
        threadgroup half w[64];
        d = dispatch[i];
        if (sgiitg == 0) {
            simdgroup_half8x8 tmp;
            simdgroup_load(tmp, &weights[as_type<uint>(d[1]) + wOff]);
            simdgroup_store(tmp, w);
        }
        // w contains
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        half myW = w[myOff];
        half wJ = as_type<short>(myW) % 15;
        myVal += wJ == myJ ? myW * d[0] : 0;
    }
    
    result[id.x] = myVal;
}
                            /*
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
  }*/

