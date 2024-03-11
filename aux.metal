//
//  aux.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 10/03/2024.
//

#include <metal_stdlib>
using namespace metal;

// rms_norm
kernel void sum_of_squares(const device half* input [[buffer(0)]],
                           device atomic_float* sum [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(sum, input[id] * input[id], memory_order_relaxed);
}

kernel void normalize_vector(device half* input [[buffer(0)]],
                             device half* output [[buffer(1)]],
                             device float* sum [[buffer(2)]],
                             const device int& count [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    float mean = sum[0]/count;
    output[id] = input[id] / sqrt(mean + 1e-6);
}

// softmax
kernel void sum_of_exps(const device half* input [[buffer(0)]],
                           device atomic_float* sum [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(sum, exp(input[id]), memory_order_relaxed);
}

kernel void softmax_add(device half* vec [[buffer(0)]],
                             device float* sum [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    vec[id] = exp(vec[id])/sum[0];
}

// dotproduct & scores
/*deploy(encoder, fname: "dot", buffers: [xq_heads[headNo], xkTokenHeads[t2][headNo]], numThreads:xq_heads[headNo].rows)
deploy(encoder, fname: "setScore", buffers:[sum, scores.ScalarAt().buffer], numThreads: 1)*/
kernel void dot(const device half* v [[buffer(0)]],
                const device half* w [[buffer(1)]],
                device atomic_float* sum [[buffer(2)]],
                uint id [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(sum, v[id]*w[id], memory_order_relaxed);
}

#define headDim 128  // llama head dim

kernel void setScore(const device float* sum [[buffer(0)]],
                     device half* target) {
    target[0] = float(sum[0]) / sqrt(float(headDim));
}
                    

/*
func softmax(_ array: inout [Float16]) {
    // Compute exponentials and sum them up
    let exps = array.map { Float16(exp(Float($0))) }
    let sumExps = exps.reduce(Float16(0.0), +)

    // Normalize each element
    for i in array.indices {
        array[i] = exps[i] / sumExps
    }
}
*/
