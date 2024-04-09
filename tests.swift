//
//  tests.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 24/03/2024.
//

import Foundation

func modelRunTests() {
  
    let modelData = Model(numLayers: 3, numExperts: 1, percentLoad: 0x10)
    let t = Tokeniser(modelData)

    //var tokens = [VectorFloat]()
    let tokens = t.embed([1, 1602, 460])

    let v = tokens[0]
    let ew = modelData.layers[0]!.w1
    let control = VectorFloat(shape:[ew.outSize])
    let test = VectorFloat(shape:[ew.outSize])

    
    print(v.str)
    print(ew.buckets.str)
    
    expertMul(v: v, by: ew, expNo: ScalarFloat(value: 0), out: control)
//    gpu.startCapture()
    bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, quant: 1)
    gpu.stopCapture()

    timeIt(repeats:1000) { i in
        let ew = modelData.layers[i % 3]!.w1
        bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, quant: 0.15)
    }
  //  gpu.startCapture()

    bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, quant: 1)
    gpu.stopCapture()

    print()
    let score = test.cosineSimilarityTo(control)
    print("\(Double(score), precision:2)", score>0.99 ? "✓" : "✗")
    print()

    
    exit(0)
    
}
