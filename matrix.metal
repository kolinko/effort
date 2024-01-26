//
//  matrix.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 26/01/2024.
//

#include <metal_stdlib>
using namespace metal;



kernel void matrixVectorMultiply(device const float *matrix [[buffer(0)]],
                                 device const float *vector [[buffer(1)]],
                                 device float *result [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    float sum = 0.0;
    int offset = id % 10000;// * 4096;

    for (int i = 0; i < 4000; ++i) {
        sum += matrix[(offset+i)] * vector[i];
    }

    result[id] = sum;


    for (int i = 0; i < 4000; ++i) {
        result[i] += sum * matrix[(offset + i)];
    }
}

