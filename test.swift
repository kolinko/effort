//
//  test.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 07/02/2024.
//

import Foundation


let numTokens = 1
let startTime = Date()

let thisToken = 0

// Q = data

for layerNo in 0...3 {
    var h = tokens[thisToken]
    
    let layer = modelData.layers[layerNo]!
    let wa = layer["attention_norm"]!
    
    let wq = layer["attention.wq"]!
    let wk = layer["attention.wk"]!
    let wv = layer["attention.wv"]!
    
    let wo = layer["attention.wo"]!
    
    let h_norm = h.rmsNorm()
    let xh = mul(vec: xh, by:wa)

    let xq = mul_col(vec: xh, by: wq)
    let xk = mul_col(vec: xh, by: wk)
    let xv = mul_col(vec: xh, by: wv)

    for t2 in 0..<thisToken {
        for headNo in 0..<numHeads {
            let sum = dot(xq, xk_Q[t2])
            scores[t2] = sum
        }
    }
    
    // add thisToken
    
    scores[thisToken] = dot(xq, qk)
    
    softmax(&scores)
    
    var output = makeArray(dims: [numHeads, headDim], value: Float16(0.0)) as! [[Float16]]
    
    for i in dim_range {
            var suma: Float16 = 0.0
            for tok2 in 0...thisToken {
                suma += scores[tok2] * xv[i]
            }
            out[headNo][i] = suma
    }
    print("computen timen \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")

    // merge heads
    var output = [Float16]()

    for headNo in 0..<numHeads {
        for i in 0..<headDim {
            output.append(out[headNo][i])
        }
    }
    
    // ffn output
    let attnOutput = Layer(from: output, using: device)
    let attnFfn = mul_col(vec: attnOutput, by: wo)
    print("compute time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")

    assert(h.test("h", mul:100, val:[0.02, -0.01, 0.01, 0.02, -0.01]))
    assert(attnFfn.test("attn", mul: 100, val:[-0.05, -0.02, -0.09, -0.07, -0.04]))
    
    add(dest: &h, by: attnFfn)
    assert(h.test("h", mul:100, val:[-0.03, -0.03, -0.07, -0.04, -0.05]))
    print("compute time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")

    let h_norm2 = h.rmsNorm()
    assert(h_norm2.test("h_norm2", mul:100, val:[-0.74, -0.69, -1.71, -0.949, -1.246]))
    let wn = layer["ffn_norm"]!
    let w1 = layer["feed_forward.w1"]!
    let w2 = layer["feed_forward.w2"]!
    let w3 = layer["feed_forward.w3"]!

    let fxn = mul(vec: h_norm2, by:wn)
    assert(fxn.test("fxn", mul:100, val:[-0.04, -0.06, -0.14, -0.07, -0.09]))
    print("compute timex \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")

    // below = 22.28-15.73 = 6ms = 1/4th of all the loop
    
    ffn(&h, fxn:fxn, w1:w1, w2:w2, w3:w3)
    print("compute time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")

    assert(h.test("h", mul:100, val:[-0.06,-0.12,-0.05,-0.09,0.01,-0.01,-0.07]))
    print("success!")

    print("compute time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")

    exit(0)
