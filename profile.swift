//
//  profile.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 23/03/2024.
//

import Foundation

func modelProfile() {
    let bucketMul = BucketMul()

    print("begin")
    let layer = modelData.layers[31]!
    let h = tokens[0]

    let buffer16 = Vector(shape:[layer.w1.outSize])
    let buffer32 = VectorFloat(shape: [layer.w1.outSize])

    gpu.eval()
    
    bucketMul.calcDispatch(v: h, weights: layer.w1, quant: 0.25)
    bucketMul.mul(v: h, by:layer.w1, out: buffer32)
    
    gpu.eval()
    print(buffer32.str())
    mpsMul(v:h, by:layer.w1, out: buffer16)
    gpu.eval()
    print(buffer16.str())
    print("cosine similarity", buffer32.cosineSimilarityTo(buffer16)[0])

    for _ in 0..<5 {
        for layerNo in 0..<32 {
            let layer = modelData.layers[layerNo]!
            bucketMul.calcDispatch(v: h, weights: layer.w1, quant: 0.25)

            bucketMul.mul(v: h, by: layer.w1, out: buffer32)
            bucketMul.mul(v: h, by: layer.w3, out: buffer32)

            mpsMul(v: h, by: layer.w1, out: buffer16)
            gpu.reEncode()
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
                bucketMul.calcDispatch(v: h, weights: layer.w1, quant: 0.25)
                bucketMul.mul(v: h, by:layer.w1, out: buffer32)
            } else {
                mpsMul(v: h, by: layer.w1, out: buffer16)
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
