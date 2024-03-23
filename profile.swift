//
//  profile.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 23/03/2024.
//

import Foundation

func modelProfile() {
    
    print("begin")
    let layer = modelData.layers[31]!
    let h = tokens[0]

    let weights = layer["feed_forward.w1"]!
    let weightBuckets = layer["feed_forward.w1.bins"]!


    let buffer16 = Vector(shape:[weights.rows])
    let buffer32 = VectorFloat(shape: [weights.rows])

    let dispatch = DynaVectorFloat(shape: [weightBuckets.rows*2])
    let dispatchSize = ScalarFloat(value: 0.0)

    calcDispatch(v: h, weights: weights, weightBuckets: weightBuckets, binsStats: layer["feed_forward.w1.bins.stats"]!,
                 dispatch: dispatch, quant: 1.0)
    gpu.eval()

    bucketMul(v: h, weightBuckets: layer["feed_forward.w1.bins"]!, weights: weights, out: buffer32, dispatch: dispatch)
    gpu.eval()
    print(buffer32.str())
    mpsMul(vector: h, weights: layer["feed_forward.w1"]!, result: buffer16)
    gpu.eval()
    print(buffer16.str())
    print("cosine similarity", buffer32.cosineSimilarityTo(buffer16)[0])
    //exit(0)

    for _ in 0..<5 {
        for layerNo in 0..<32 {
            let layer = modelData.layers[layerNo]!
            let weightBuckets = layer["feed_forward.w1.bins"]!
            let weightBuckets3 = layer["feed_forward.w3.bins"]!
            let weights = layer["feed_forward.w1"]!
            
            bucketMul(v: h, weightBuckets: weightBuckets, weights: weights, out: buffer32, dispatch: dispatch)
            bucketMul(v: h, weightBuckets: weightBuckets3, weights: weights, out: buffer32, dispatch: dispatch)

            mpsMul(vector: h, weights: weights, result: buffer16)
        }
    }
    gpu.eval()
    print("warmed up, redoing now")


    var numLayersProf = 32
    var repeats=30
    let captureGPU = false
    let mine = true

    if captureGPU {
        repeats = 5
        numLayersProf = 5
        gpu.startCapture(cond:captureGPU)
        gpu.eval()
    }
    startTime = Date()

    for _ in 0..<repeats*4 {
        for layerNo in 0..<numLayersProf {
            let layer = modelData.layers[layerNo]!
            if mine {
                let weightBuckets = layer["feed_forward.w1.bins"]!
                bucketMul(v: h, weightBuckets: weightBuckets, weights: weights, out: buffer32, dispatch: dispatch)
            } else {
                mpsMul(vector: h, weights: layer["feed_forward.w1"]!, result: buffer16)
            }
        }
    }
    print("prep time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
    startTime = Date()

    print("eval")
    gpu.eval()
    print("total time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
    print("\ncycle time \(Date().timeIntervalSince(startTime)*1000/Double(repeats), precision: 2) ms\n")

    gpu.stopCapture()
    print(buffer16[0])
    
}
