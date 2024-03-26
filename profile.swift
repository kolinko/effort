//
//  profile.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 23/03/2024.
//

import Foundation

func modelProfile(captureGPU: Bool = false, mine: Bool = true) {
    let bucketMul = BucketMul.shared

    print("begin")
    let layer = modelData.layers[31]!
    let h = tokens[0]

    let buffer16 = Vector(shape:[layer.w1.outSize])
    let buffer32 = VectorFloat(shape: [layer.w1.outSize])
    
    bucketMul.calcDispatch(v: h, weights: layer.w1, quant: 0.25)
    bucketMul.mul(by:layer.w1, out: buffer32)
    mpsMul(v:h, by:layer.w1, out: buffer16)
    gpu.eval()
    print(buffer32.str())
    print(buffer16.str())
    print("cosine similarity", buffer32.cosineSimilarityTo(buffer16)[0])

    let hx = buffer32.asFloat16Vector()
    let buffer16x = Vector(shape:[layer.w2.outSize])
    let buffer32x = VectorFloat(shape: [layer.w2.outSize])

    mpsMul(v:hx, by:layer.w2, out: buffer16x)
    gpu.eval()
    bucketMul.calcDispatch(v: hx, weights: layer.w2, quant: 1)
    bucketMul.mul(by:layer.w2, out: buffer32x)
    gpu.eval()
    print(buffer32x.str())
    print(buffer16x.str())
    print("cosine similarity", buffer32x.cosineSimilarityTo(buffer16x)[0])

    gpu.eval()
    
    for _ in 0..<5 {
        for layerNo in 0..<32 {
            let layer = modelData.layers[layerNo]!
            bucketMul.calcDispatch(v: h, weights: layer.w1, quant: 0.25)

            bucketMul.mul(by: layer.w1, out: buffer32)
            bucketMul.mul(by: layer.w3, out: buffer32)

            mpsMul(v: h, by: layer.w1, out: buffer16)
            gpu.reEncode()
        }
    }
    gpu.eval()
    print("warmed up, redoing now")


    var numLayersProf = 32
    var repeats=30

    if captureGPU {
        repeats = 5
        numLayersProf = 5
        gpu.startCapture(cond:captureGPU)
        gpu.eval()
    }

    var startTime = Date()

//    bucketMul.calcDispatch(v: hx, weights: layer.w2, quant: 0.10)
    bucketMul.calcDispatch(v: h, weights: layer.w1, quant: 0.10)

    for _ in 0..<repeats*4 {
        for layerNo in 0..<numLayersProf {
            let layer = modelData.layers[layerNo]!
            if mine {
//                bucketMul.calcDispatch(v: hx, weights: layer.w2, quant: 0.25)
//                bucketMul.mul(v: hx, by:layer.w2, out: buffer32x)
//                bucketMul.calcDispatch(v: h, weights: layer.w1, quant: 0.25)
                bucketMul.mul(by:layer.w1, out: buffer32)
            } else {
                
                mpsMul(v: hx, by: layer.w2, out: buffer16x)
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
    print(buffer16.str())    
}
