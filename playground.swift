//
//  tests.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 24/03/2024.
//

import Foundation

func modelRunTests() {
  
    let modelData = Model(numLayers: 11, numExperts: 1, percentLoad: 0x10)
    let t = Tokeniser(modelData)

    //var tokens = [VectorFloat]()
    let tokens = t.embed([1, 1602, 460])
/*
    for i in 0..<32 {
        let ew = modelData.layers[i]!.wq
        print(ew.core!.str)
    }
    exit(0)
*/
    
    let v = TensorLoader.loadVec("xq_broken") //tokens[0]
    let ew = modelData.layers[10]!.wq
    let control = basicMul(v: v, by: ew.core!) //VectorFloat(shape:[ew.outSize])
    let test = VectorFloat(shape:[ew.outSize])
    
    print(v.str)
    print(ew.buckets.str)
    print()
    print(control.str)
    //gpu.startCapture()
    for q in [0.01, 0.02, 0.03] {
        test.zero()
        expertMul(v: v, by: ew, out: test, quant: q)
        print(test.str)
        assert(!test.hasNan)
        //gpu.stopCapture()
        //        print()
        let score = test.cosineSimilarityTo(control)
        print("\(Int(q*100))%: \(Double(score), precision:5)", score>0.99 ? "✓" : "✗")
        //        print()
    }
    exit(0)
    
    
    /*expertMul(v: v, by: ew, expNo: ScalarFloat(value: 0), out: control)
//    gpu.startCapture()
    bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, quant: 1)
    gpu.stopCapture()

    timeIt(repeats:1000) { i in
        let ew = modelData.layers[i % 3]!.w1
        expertMul(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, quant: 1)
    }
    print()
    timeIt(repeats:1000) { i in
        let ew = modelData.layers[i % 3]!.w1
        bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, quant: 1)
    }

    //    gpu.startCapture()

    bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, quant: 1)
    gpu.stopCapture()

    print()
    let score = test.cosineSimilarityTo(control)
    print("\(Double(score), precision:5)", score>0.99 ? "✓" : "✗")
    print()

    
    exit(0)
    */
}

/*func exampleH -> VectorFloat {
    

    
}*/
