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
    /*
    for i in 0..<count {
        let complexNum = (self[2 * i], self[2 * i + 1])
        let result = multiplyComplex(complexNum, complexArray[i])
        self[2*i] = result.0
        self[2*i+1] = result.1
    }
     */
    /*
    let (a, b) = num1
    let (c, d) = num2
    return (a * c - b * d, a * d + b * c)*/

    
}

// cosine similarity

//gpu.deploy("cosinePrecalc", buffers: [self, vec, dotBuffer, normABuffer, normBBuffer], threadCount: self.rows)
//gpu.deploy("cosineCalc", buffers: [dotBuffer, normABuffer, normBBuffer], threadCount: 0)
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


/*
deploy(encoder, fname: "sumScores", buffers:[scoresMatrix, xvTokenMatrix, outMatrix], ints: [numTokens], threadCount: numHeads*headDim)
encoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

let scoresMatrix = gpuConsolidate(vecList: scores).asVectorList()
let xvTokenMatrix = gpuConsolidate(vecList: xvToken).asVectorList()

for headNo in 0..<numHeads {
    for i in 0..<headDim {
        var suma: Float16 = 0.0
        for tok2 in 0...thisToken {
            suma += scoresMatrix[headNo][tok2] * xvTokenMatrix[tok2][headNo*headDim+i]
        }
        out[headNo][i] = suma
    }
}*/

/*deploy(encoder, fname: "dot", buffers: [xq_heads[headNo], xkTokenHeads[t2][headNo]], numThreads:xq_heads[headNo].rows)
deploy(encoder, fname: "setScore", buffers:[sum, scores.ScalarAt().buffer], numThreads: 1)*/
kernel void dot(const device half* v [[buffer(0)]],
                const device half* w [[buffer(1)]],
                device atomic_float* sum [[buffer(2)]],
                uint id [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(sum, v[id]*w[id], memory_order_relaxed);
}


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
