//
//  matrix.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 26/01/2024.
//

#include <metal_stdlib>
using namespace metal;


kernel void mul_col_4096(device const half *matrix [[buffer(0)]],
                    device const half *vector [[buffer(1)]],
                    device half *result [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    half sum = 0.0;
    int offset = id * 4096;
    for (int i = 0; i < 4096; i++) {
        sum += matrix[(offset+i)] * vector[i];
    }

    result[id] = sum;
}


kernel void mul_col_11008(device const half *matrix [[buffer(0)]],
                    device const half *vector [[buffer(1)]],
                    device half *result [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    half sum = 0.0;
    int offset = id * 11008;
    for (int i = 0; i < 11008; i++) {
        sum += matrix[(offset+i)] * vector[i];
    }

    result[id] = sum;
}

