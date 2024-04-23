//
//  tests.swift
//  effort
//
//  Created 24/03/2024.
//

/*
  
 For quick testing of various functionalities.
 
 */

import Foundation

func goPlayground() {
  
    let modelData = Model(numLayers: 13, numExperts: 1, percentLoad: 0x10)
    let t = Tokeniser(modelData)
    
    let v = TensorLoader.loadVec("xq_broken") //tokens[0]
    let ew = modelData.layers[12]!.wq
    let control = VectorFloat(shape:[ew.outSize])
//    timeIt(repeats:2000) { i in
    bucketMul(v: v, by: ew, expNo: ScalarFloat(value: 0), out: control, effort: 1.0)
    gpu.eval()
    let ts = TensorSaver(path: ".", model: "q4data2")
    ts[0]["v"] = v
    ts[0]["core"] = ew.core!
    ts[0]["control"] = control
    ts.save()
    
    gpu.stopCapture()

    //    }
    let test = VectorFloat(shape:[ew.outSize])
    let q = 1.0

        gpu.startCapture()
        bucketMulWild(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, effort: q)
    gpu.stopCapture()
//    }
    let score = test.cosineSimilarityTo(control)

    print("\(perc: q): \(Double(score), precision:5)", score>0.99 ? "✓" : "✗")

    exit(0)
    //    let control = basicMul(v: v, by: ew.core!) //
    
    /*
    print(v.str)
    print(ew.buckets.str)
    print()
    print(control.str)
    //gpu.startCapture()
    for q in [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0] {
        expertMul(v: v, by: ew, out: test, effort: q)
        assert(!test.hasNan)
        //gpu.stopCapture()
        //        print()
        let score = test.cosineSimilarityTo(control)
        print("\(Int(q*100))%: \(Double(score), precision:5)", score>0.99 ? "✓" : "✗")
        //        print()
    }*/

    bucketMul(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, effort: q)
    print("\(Int(q*100))%: \(Double(score), precision:5)", score>0.99 ? "✓" : "✗")

    exit(0)
    
    
    /*expertMul(v: v, by: ew, expNo: ScalarFloat(value: 0), out: control)
//    gpu.startCapture()
    bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, effort: 1)
    gpu.stopCapture()

    timeIt(repeats:1000) { i in
        let ew = modelData.layers[i % 3]!.w1
        expertMul(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, effort: 1)
    }
    print()
    timeIt(repeats:1000) { i in
        let ew = modelData.layers[i % 3]!.w1
        bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, effort: 1)
    }

    //    gpu.startCapture()

    bucketMulFast(v: v, by: ew, expNo: ScalarFloat(value: 0), out: test, effort: 1)
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
