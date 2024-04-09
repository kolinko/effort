//
//  tests.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 24/03/2024.
//

import Foundation

func modelRunTests() {
  
    let modelData = Model(numLayers: 1, numExperts: 2, percentLoad: percentLoad)
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
    expertMul(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test)
//    gpu.stopCapture()

    timeIt(repeats:1000) { _ in
        expertMul(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, quant: 1)
    }

    print()
    let score = test.cosineSimilarityTo(control)
    print("\(Double(score), precision:2)", score>0.99 ? "✓" : "✗")
    print()

    
    exit(0)
    
}
