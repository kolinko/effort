//
//  profile.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 23/03/2024.
//

import Foundation

func timeIt(repeats: Int = 10000, withCapture: Bool = false, _ closure: (Int) -> Void) {
    print("profiling, reps \(repeats)...")
    var goTime = Date()
    gpu.startCapture(cond: withCapture)
    // warmup loop & in case of capture - just the roop
    for i in 0..<10 {
        closure(i)
    }
    if withCapture {
        gpu.stopCapture()
    }
    gpu.eval()
    
    for i in 0..<repeats {
        closure(i)
    }
    print("prep time \(Date().timeIntervalSince(goTime)*1000, precision: 2) ms")
    goTime = Date()
    gpu.eval()
    print("final eval time \(Date().timeIntervalSince(goTime)*1000, precision: 2) ms")
    let epl = Date().timeIntervalSince(goTime)*1000/Double(repeats)
    print("eval per loop \(epl, precision: 2) ms")
    print("persec \(Double(repeats) / Date().timeIntervalSince(goTime), precision: 2) runs")
    print()
    print("tpt \(epl*1*4*32, precision: 2) ms")
    print("spd \(1000/(epl*1*4*32), precision: 2) tps")

}

// 16MB. With a read speed of 300GB/s it should have a pace of 18750/sec.
// so 1875 iters would be 100ms

// 4096*14336 = 58 MB/s, 15945*896*2 = 28MB/s
/*
layer.w1.buckets.shape = [21, 4096, 4096]
let ml = layer.w1.buckets.asMatrixList()
let in16 = attnOutput.asFloat16()
let out16 = attnOutput.asFloat16()
timeIt(18700) { i in
        //basicMul(v: attnOutput, by: ml[i % 21], out: attnFfnOut)// layer.wo.core, out: attnFfnOut)
        mpsMul(v: in16, by: ml[i % 21], out: out16)
}
*/

/*
func modelProfile(captureGPU: Bool = false, mine: Bool = true) {
    let bucketMul = BucketMul.shared

    print("begin")
    let layer = modelData.layers[31]!
    let h = tokens[0]

    let buffer16 = Vector(shape:[layer.w1.outSize])
    let buffer32 = VectorFloat(shape: [layer.w1.outSize])
    
    bucketMul.calcDispatch(v: h, weights: layer.w1, effort: 0.25)
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
    bucketMul.calcDispatch(v: hx, weights: layer.w2, effort: 1)
    bucketMul.mul(by:layer.w2, out: buffer32x)
    gpu.eval()
    print(buffer32x.str())
    print(buffer16x.str())
    print("cosine similarity", buffer32x.cosineSimilarityTo(buffer16x)[0])

    gpu.eval()
    
    for _ in 0..<5 {
        for layerNo in 0..<32 {
            let layer = modelData.layers[layerNo]!
            bucketMul.calcDispatch(v: h, weights: layer.w1, effort: 0.25)

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

//    bucketMul.calcDispatch(v: hx, weights: layer.w2, effort: 0.10)
    bucketMul.calcDispatch(v: h, weights: layer.w1, effort: 0.10)

    for _ in 0..<repeats*4 {
        for layerNo in 0..<numLayersProf {
            let layer = modelData.layers[layerNo]!
            if mine {
//                bucketMul.calcDispatch(v: hx, weights: layer.w2, effort: 0.25)
//                bucketMul.mul(v: hx, by:layer.w2, out: buffer32x)
//                bucketMul.calcDispatch(v: h, weights: layer.w1, effort: 0.25)
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
*/
