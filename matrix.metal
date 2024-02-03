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
        sum += matrix[(offset + i)] * vector[i];
    }

    result[id] = sum;
     
}

#define outer_count 4096

//constant int outer_count = 4096;

kernel void internal(device const half *fxn [[buffer(0)]],
                     device const half *w1 [[buffer(1)]],
                     device const half *w3 [[buffer(2)]],
                     device half *result [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    half x1 = 0.0;
    half x2 = 0.0;
    half x3 = 0.0;
    int offset = id*outer_count;
    
    for (int i = 0; i<outer_count; i++) {
        half x = fxn[i];
        x1 += x * w1[offset + i];
        x3 += x * w3[offset + i];
    }
    x2 = x3 * x1 / (1 + exp(-x1));
    
    result[id] = x2;
}

kernel void second (device const half *matrix [[buffer(0)]],
                    device const half *vector [[buffer(1)]],
                    device half *result [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    half sum = 0.0;
    int offset = id * 11008;
    for (int i = 0; i < 11008; i++) {
        sum += matrix[(offset + i)] * vector[i];
    }

    result[id] += sum;
}



kernel void internal2(device const half *fxn [[buffer(0)]],
                     device const half *w1 [[buffer(1)]],
                      device const half *w2 [[buffer(2)]],
                      device const half *w3 [[buffer(3)]],
                      device half *result [[buffer(4)]],
                      device half *h [[buffer(5)]],
                     uint id [[thread_position_in_grid]]) {
    half x1 = 0.0;
    half x2 = 0.0;
    half x3 = 0.0;

    int offset = id*outer_count;
    
    for (int i = 0; i<outer_count; i++) {
        half x = fxn[i];
        x1 += x * w1[offset + i];
        x3 += x * w3[offset + i];
    }
    x2 = x3* x1 / (1 + exp(-x1));
    
    result[id] = x2;
    
    
    if (id >= outer_count) {
        return;
    };

    // insert barrier
    
    half sum = 0.0;
    offset = id * 11008;
    for (int i = 0; i < 11008; i++) {
        sum += w2[(offset + i)] * result[i];
    }

    h[id] += sum;
}

/*
kernel void ffn(device half *h [[buffer(0)]],
                device const half *fxn [[buffer(1)]],
                device const half *w1 [[buffer(2)]],
                device const half *w2 [[buffer(3)]],
                device const half *w3 [[buffer(4)]],
                uint id [[thread_position_in_grid]]) {
    
}
                _ h: inout Layer, fxn: Layer, w1: Layer, w2: Layer, w3: Layer) {
    let outerDim = 4096
    let innerDim = 11008
    assert(w1.shape==[11008, 4096])
    assert(w2.shape==[4096, 11008])
    assert(w3.shape==[11008, 4096])
    assert(fxn.shape==[4096])

    let fx1 = mul_col(vec: fxn, by: w1)
    let fx3 = mul_col(vec: fxn, by: w3)
    assert(fx1.shape == [11008])
    assert(fx3.shape == [11008])

    //    x = ((x1 / (1.0 + np.exp(-x1))) * x3
    var x = [(Float16)]()
    for i in 0..<fx3.rows {
        let val: Double = Double(fx1[i])/(1+exp(Double(-fx1[i]))) * Double(fx3[i])
        x.append(Float16(val))
    }

    let fx = Layer(from: x, using: device)
    let fx2 = mul_col(vec:fx, by: w2)
    assert(fx2.shape==[4096])

    add(dest: &h, by: fx2)
}
*/
