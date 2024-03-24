//
//  aux.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 10/03/2024.
//

#include <metal_stdlib>
using namespace metal;


kernel void strictDiff(const device half* a [[buffer(0)]],
                       const device half* b [[buffer(1)]],
                           device atomic_float* diff [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    if (a[id] != b[id]) {
        atomic_fetch_add_explicit(diff, 1, memory_order_relaxed);
    }
}

// rms norm

kernel void rms_norm(device half* input [[buffer(0)]],
                             device half* output [[buffer(1)]],
                             const device int& count [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    float sum = 0;
    for (int i=0; i<count; i++) {
        sum += float(input[i]) * float(input[i]);
    }
    output[id] = input[id] / sqrt(sum/count + 1e-6);
}

/* unused below ?*/

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
    vec[id] = exp(float(vec[id]))/sum[0];
}
// simple ones
kernel void mul_vec(const device half* v [[buffer(0)]],
                const device half* w [[buffer(1)]],
                device half* out [[buffer(2)]],
                uint id [[thread_position_in_grid]]) {
    out[id] = v[id]*w[id];
}

kernel void add_vec(const device half* v [[buffer(0)]],
                const device half* w [[buffer(1)]],
                device half* out [[buffer(2)]],
                uint id [[thread_position_in_grid]]) {
    out[id] = v[id]+w[id];
}

kernel void mul_complex(device half2* v [[buffer(0)]],
                        const device half2* comp [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    
    half2 num1 = v[id];
    half2 num2 = comp[id];
    
    half a = num1.x;
    half b = num1.y;
    half c = num2.x;
    half d = num2.y;
    
    half2 out;
    out.x = a * c - b * d;
    out.y = a * d + b * c;
    
    v[id] = out;    
}

// cosine similarity
kernel void floatToHalf(const device float *inVec,
                        device half *outVec,
                        uint id [[thread_position_in_grid]]) {
    outVec[id] = inVec[id];
}

kernel void cosinePrecalc(const device float *A,
                                   const device half *B,
                                   device atomic_float *dotProduct,
                                   device atomic_float *magnitudeA,
                                   device atomic_float *magnitudeB,
                                   uint id [[thread_position_in_grid]]) {
    // Compute dot product and magnitudes for cosine similarity
    atomic_fetch_add_explicit(dotProduct, A[id] * B[id], memory_order_relaxed);
    atomic_fetch_add_explicit(magnitudeA, A[id] * A[id], memory_order_relaxed);
    atomic_fetch_add_explicit(magnitudeB, B[id] * B[id], memory_order_relaxed);
}

kernel void cosinePrecalc16(const device half *A,
                                   const device half *B,
                                   device atomic_float *dotProduct,
                                   device atomic_float *magnitudeA,
                                   device atomic_float *magnitudeB,
                                   uint id [[thread_position_in_grid]]) {
    // Compute dot product and magnitudes for cosine similarity
    atomic_fetch_add_explicit(dotProduct, A[id] * B[id], memory_order_relaxed);
    atomic_fetch_add_explicit(magnitudeA, A[id] * A[id], memory_order_relaxed);
    atomic_fetch_add_explicit(magnitudeB, B[id] * B[id], memory_order_relaxed);
}

kernel void cosineCalc(device float *dotProduct,
                       device float *magnitudeA,
                       device float *magnitudeB) {
    dotProduct[0] = dotProduct[0]/(sqrt(magnitudeA[0]) * sqrt(magnitudeB[0]));
}


// dotproduct & scores
kernel void memcpy(const device half* src [[buffer(0)]],
                device half* dst [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) {
    dst[id] = src[id];
}

#define numHeads 32
#define headDim 128  // llama head dim
#define numDims numHeads*headDim

kernel void sumScores(const device half* scores [[buffer(0)]],
                      const device half* xvToken [[buffer(1)]],
                      device half* out [[buffer(2)]],
                      const device int& numTokens [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
    float suma = 0.0;
    uint headNo = id / headDim;
    for (int tok2 = 0; tok2 < numTokens; tok2++) {
        suma += scores[headNo*numTokens + tok2] * xvToken[tok2*numDims + id];
    }
    out[id] = suma;
}

kernel void dot(const device half* v [[buffer(0)]],
                const device half* w [[buffer(1)]],
                device atomic_float* sum [[buffer(2)]],
                uint id [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(sum, v[id]*w[id], memory_order_relaxed);
}


kernel void setScore(const device float* sum [[buffer(0)]],
                     device half* target) {
    target[0] = float(sum[0]) / sqrt(float(headDim) + 1e-6);
}
                    
