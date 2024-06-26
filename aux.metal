//
//  aux.metal
//  effort
//
//  Created 10/03/2024.
//

#include <metal_stdlib>
using namespace metal;

#define numHeads 32
#define headDim 128  // llama head dim
#define numDims numHeads*headDim

kernel void touch(device half* v[[buffer(0)]],
                  device int* bsScalar[[buffer(1)]],
                  const device int& vSize[[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    bsScalar[0] += 123*v[uint(id*bsScalar[0]) % vSize];
    bsScalar[0] += 123*v[uint(id*bsScalar[0]) % vSize];
    v[uint(id*bsScalar[0]) % vSize] += 1e-10;

}

//gpu.deploy("hasNan\(bitSize)", buffers: [self, tmpScalar], threadCount: self.count)
kernel void hasNan16(device half* v[[buffer(0)]],
                     device atomic_float* out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    if (isnan(v[id])) {
        atomic_fetch_add_explicit(out, 1, memory_order_relaxed);
    }
}

kernel void hasNan32(device float* v[[buffer(0)]],
                     device atomic_float* out [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    if (isnan(v[id])) {
        atomic_fetch_add_explicit(out, 1, memory_order_relaxed);
    }
}


kernel void zero16(device half* v[[buffer(0)]],
                    uint id [[thread_position_in_grid]]) {
    v[id] = 0;
}

kernel void zero32(device float* v[[buffer(0)]],
                   uint id [[thread_position_in_grid]]) {
    v[id] = 0;
}


kernel void neg16(device half* v[[buffer(0)]],
                    uint id [[thread_position_in_grid]]) {
    v[id] = -v[id];
}

kernel void neg32(device float* v[[buffer(0)]],
                   uint id [[thread_position_in_grid]]) {
    v[id] = -v[id];
}

kernel void mulScalar16x16(device half* v [[buffer(0)]],
                const device half* s [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    v[id] = v[id]*s[0];
}

kernel void mulScalar32x16(device float* v [[buffer(0)]],
                const device half* s [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    v[id] = v[id]*s[0];
}

kernel void mulScalar16x32(device half* v [[buffer(0)]],
                            const device float* s [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    v[id] = v[id]*s[0];
}

kernel void mulScalar32x32(device float* v [[buffer(0)]],
                const device float* s [[buffer(1)]],
                uint id [[thread_position_in_grid]]) {
    v[id] = v[id]*s[0];
}

kernel void add16(const device half* v [[buffer(0)]],
                const device half* w [[buffer(1)]],
                device half* out [[buffer(2)]],
                uint id [[thread_position_in_grid]]) {
    out[id] = v[id]+w[id];
}

kernel void add32(const device float* v [[buffer(0)]],
                const device float* w [[buffer(1)]],
                device float* out [[buffer(2)]],
                uint id [[thread_position_in_grid]]) {
    out[id] = v[id]+w[id];
}


kernel void strictDiff(const device half* a [[buffer(0)]],
                       const device half* b [[buffer(1)]],
                           device atomic_float* diff [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    if (a[id] != b[id]) {
        atomic_fetch_add_explicit(diff, 1, memory_order_relaxed);
    }
}

// rms norm
kernel void rmsNorm32fast(device float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             uint count [[threads_per_grid]],
                             uint id [[thread_position_in_grid]],
                              ushort tiisg [[thread_index_in_simdgroup]],
                              ushort sgiitg [[simdgroup_index_in_threadgroup]],
                              ushort sgptg [[simdgroups_per_threadgroup]]) {

    threadgroup float temp[32] = {0};
    threadgroup float full_sum = 0;
    float sum = 0;
    
    // reduce over threads in the group
    const uint mul = 4;
    const uint begin = id*mul;
    const uint end = id*mul+mul;
    const uint numEls = count * mul;
    
    for (uint i = begin; i<end; i++) {
        sum += float(input[i])*float(input[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = simd_sum(sum);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        temp[sgiitg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgiitg == 0) {
        sum = temp[tiisg];
        full_sum = simd_sum(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = begin; i<end; i++) {
        output[i] = input[i] / sqrt(float(full_sum)/float(numEls) + 1e-5);
    }
}

kernel void rmsNorm32(device float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             uint count [[threads_per_grid]],
                             uint id [[thread_position_in_grid]]) {
    float sum = 0;
    for (uint i=0; i<count; i++) {
        sum += float(input[i]) * float(input[i]);
    }

    output[id] = input[id] / sqrt(sum/count + 1e-6);
  //  output[0] = 10+sum;
}

/* unused below ?*/
/*
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
}*/

kernel void sum_of_exps32_mx(const device float* input [[buffer(0)]],
                          device atomic_float* sum [[buffer(1)]],
                          device const uint &numCols [[buffer(2)]],
                          uint2 id [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(&sum[id.y], exp(input[id.x+id.y*numCols]), memory_order_relaxed);
}

kernel void softmax_add32_mx(device float* vec [[buffer(0)]],
                             device float* sum [[buffer(1)]],
                             device const uint &numCols [[buffer(2)]],
                             uint2 id [[thread_position_in_grid]]) {
    
    vec[id.x+id.y*numCols] = exp(vec[id.x+id.y*numCols]) / sum[id.y];
}

// softmax
kernel void sum_of_exps32(const device float* input [[buffer(0)]],
                          device atomic_float* sum [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
//    atomic_max_explicit(sum, input[id], memory_order_relaxed);
    atomic_fetch_add_explicit(sum, exp(input[id]), memory_order_relaxed);
//    atomic_fetch_max_explicit(sum, exp(input[id]), memory_order_relaxed);
}

kernel void softmax_add32(device float* vec [[buffer(0)]],
                          device float* sum [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    
    vec[id] = exp(vec[id]) / sum[0];
}


// modelling_mixtral.py from transformers, def rotate_half & apply_rotary_pos_emb
kernel void rope_mx(const device float* m [[buffer(0)]],
                    const device float2* comp [[buffer(1)]],
                    device float* out [[buffer(2)]],

                    const device uint &numEls [[buffer(3)]],
                    uint2 id [[thread_position_in_grid]]) {
    uint myOff = id.y * numEls + id.x;
    
    if (id.x < numEls/2) {
        out[myOff] = m[myOff] * comp[id.x].x - m[myOff + numEls/2] * comp[id.x].y;
    } else {
        out[myOff] = m[myOff] * comp[id.x].x + m[myOff - numEls/2] * comp[id.x].y;
    }
}

kernel void mulComplex32_mx(device float2* m [[buffer(0)]],
                        const device float2* comp [[buffer(1)]],
                        const device uint &numCols [[buffer(2)]],
                        uint2 id [[thread_position_in_grid]]) {
    
    float a = m[id.x+id.y*numCols/2].x;
    float b = m[id.x+id.y*numCols/2].y;
    float c = comp[id.x].x;
    float d = comp[id.x].y;
    
    float2 out;
    out.x = a * c - b * d;
    out.y = a * d + b * c;
    
    m[id.x+id.y*numCols/2] = out;
    
}

kernel void repeat4x32(const device float* v [[buffer(0)]],
                   device float* out [[buffer(1)]],
                   uint2 id [[thread_position_in_grid]],
                   uint2 tpg [[threads_per_grid]]) {
    
    for(int i = 0; i<4; i++) {
        out[id.x+i*tpg.x+id.y*tpg.x*4] = v[id.x+(id.y*tpg.x)];
    }
    
}

kernel void mulVec16by16(const device half* v [[buffer(0)]],
                         const device half* w [[buffer(1)]],
                         device half* out [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    out[id] = v[id]*w[id];
}

kernel void mulVec32by16(const device float* v [[buffer(0)]],
                         const device half* w [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    out[id] = v[id]*w[id];
}



// cosine similarity
kernel void floatToHalf(const device float *inVec,
                        device half *outVec,
                        uint id [[thread_position_in_grid]]) {
    outVec[id] = inVec[id];
}

kernel void halfToFloat(const device half *inVec,
                        device float *outVec,
                        uint id [[thread_position_in_grid]]) {
    outVec[id] = inVec[id];
}



kernel void cosinePrecalc32(const device float *A,
                            const device float *B,
                            device atomic_float *dotProduct,
                            device atomic_float *magnitudeA,
                            device atomic_float *magnitudeB,
                            uint id [[thread_position_in_grid]]) {
    
    // Compute dot product and magnitudes for cosine similarity
    atomic_fetch_add_explicit(dotProduct, A[id] * B[id], memory_order_relaxed);
    atomic_fetch_add_explicit(magnitudeA, A[id] * A[id], memory_order_relaxed);
    atomic_fetch_add_explicit(magnitudeB, B[id] * B[id], memory_order_relaxed);
    
}


kernel void cosineCalc32(device float *dotProduct,
                       device float *magnitudeA,
                       device float *magnitudeB) {
    dotProduct[0] = dotProduct[0]/(sqrt(magnitudeA[0]) * sqrt(magnitudeB[0]));
}


// dotproduct & scores
kernel void memcpy16(const device half* src [[buffer(0)]],
                     device half* dst [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    dst[id] = src[id];
}

kernel void memcpy32(const device float* src [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    dst[id] = src[id];
}

kernel void memcpyBig16(const device half* src [[buffer(0)]],
                     device half* dst [[buffer(1)]],
                     device const uint& batchSize [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    uint begin = id*batchSize;
    uint end = begin + batchSize;
    for (uint i = begin; i<end; i++) {
        dst[i] = src[i];
    }
}

kernel void setVal(device uint* result [[buffer(0)]],
                        device const uint& number [[buffer(1)]],

                     uint id [[thread_position_in_grid]]) {
    result[0] = number;

}

kernel void convertBF16X(device bfloat *src [[buffer(0)]],
                        device half *dst [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    half x = half(src[id]);
    dst[id] = x;
}

//gpu.deploy("fetchRow16to32", buffers: [rowNum, self, out], threadCount: self.rows)
kernel void fetchRow16to32(const device int* rowNo,
                           const device half* src,
                           device float* dst,
                           uint id [[thread_position_in_grid]],
                           uint cols [[threads_per_grid]]) {
    
    dst[id] = src[rowNo[0]*cols + id];
}


kernel void memcpyBig32(const device float* src [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                        device const uint& batchSize [[buffer(2)]],

                     uint id [[thread_position_in_grid]]) {
    uint begin = id*batchSize;
    uint end = begin + batchSize;
    for (uint i = begin; i<end; i++) {
        dst[i] = src[i];
    }

}


kernel void sumScores32(const device float* scores [[buffer(0)]],
                      const device float* xvToken [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      const device int& numTokens [[buffer(3)]],
                        const device int& scoresCols [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    float suma = 0.0;
    uint headNo = id / headDim;
    for (int tok2 = 0; tok2 < numTokens; tok2++) {
        suma += scores[headNo*scoresCols + tok2] * xvToken[tok2*numDims + id];
    }
    out[id] = suma;
}



// setScore

kernel void dotSetScore2(const device float* xqHeads [[buffer(0)]],
                        const device float* xkTokenHeads [[buffer(1)]],
                        device float* scores [[buffer(2)]],
                        ushort3 id [[thread_position_in_grid]],
                        ushort tiisg [[thread_index_in_simdgroup]],
                        ushort sgiitg [[simdgroup_index_in_threadgroup]],
                        ushort sgptg [[simdgroups_per_threadgroup]],
                        ushort3 tpg [[threads_per_grid]]
                                //threadgroup size == thread count!
                        ) {
    
    short dimSize = tpg.x;
    assert(dimSize == 127);

    short headNo = id.z;
//    short headsCount = tpg.z;
//    assert(headsCount == numHeads);
    
    short t2 = id.y;
    short numTokens = tpg.y;
    
    device float* scoresOut = &scores[headNo * numTokens + t2];
    const device float* v = &xqHeads[headNo * dimSize];
    const device float* w = &xkTokenHeads[t2*numHeads*dimSize + headNo*dimSize];
    
    threadgroup half temp[32] = {0};
    float sum = 0;
    
    // reduce over threads in the group
    uint begin = id.x;
    uint end = (id.x+1);
    for (uint i = begin; i<end; i++) {
        sum += float(v[i])*float(w[i]);
    }
    sum = simd_sum(sum);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        temp[sgiitg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (id.x==0) {
        sum = 0;
        for (int i=0; i<sgptg; i++) {
            sum += temp[i];
        }
        assert(tiisg == 0);
        assert(sgiitg == 0);
        *scoresOut = sum / sqrt(float(headDim));
    }
}

/*
func calcScores2(xq_heads: [VectorFloat], xkTokenHeads: Matrix3DFloat) -> [VectorFloat] {
    let numTokens = xkTokenHeads.slices
    let scores = MatrixFloat(shape: [numHeads, numTokens])

    for t2 in 0..<numTokens {
        for headNo in 0..<numHeads {
            assert(xq_heads[headNo].rows == 128, "not tested/implemented for other values.");
            gpu.deploy("dotSetScore2",
                       buffers: [xq_heads[headNo], xkTokenHeads[t2][headNo], scores.scalarAt(headNo, t2)],
                       threadCount:[128, numHeads, numTokens],
                       threadGroupSize: [128, 1, 1])
        }
    }
    
    return scores.asVectorList()
}*/

kernel void dotSetScore32(const device float* v [[buffer(0)]],
                        const device float* w [[buffer(1)]],
                        device float* target [[buffer(2)]],
                        const device int& chunkSize [[buffer(3)]],
                        ushort id [[thread_position_in_grid]],
                        ushort tiisg [[thread_index_in_simdgroup]],
                        ushort sgiitg [[simdgroup_index_in_threadgroup]],
                        ushort sgptg [[simdgroups_per_threadgroup]]
                                //threadgroup size == thread count!
                        ) {
    threadgroup half temp[32] = {0};
    float sum = 0;
    uint begin = id*chunkSize;
    uint end = (id+1)*chunkSize;
    for (uint i = begin; i<end; i++) {
        sum += float(v[i])*float(w[i]);
    }
    sum = simd_sum(sum);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        temp[sgiitg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (id==0) {
        sum = 0;
        for (int i=0; i<sgptg; i++) {
            sum += temp[i];
        }
        assert(tiisg == 0);
        assert(sgiitg == 0);
        target[0] = sum / sqrt(float(headDim));
    }
}


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
                    
