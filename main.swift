//
//  main.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 24/01/2024.
//
import os
import Foundation
import Metal
import simd


let log = OSLog(subsystem: "com.kolinko", category: "Performance")
 
let gpu = Gpu()
let gpu2 = Gpu()
print("loading")

let modelData = Model(from: "shape.json")

var tokens = [VectorFloat]()
let tokIds = [1, 1602, 460] // "How are"
let t = Tokeniser()

let tokEmbeddings = modelData.tokEmbeddings.asVectorList()
for t in tokIds {
    tokens.append(tokEmbeddings[t].asFloat32())
}
os_signpost(.end, log: log, name: "Loading")

let headDim = 128  // Example head dimension
let numHeadsKV = 8
let numHeads = 32
let kvRepeats : Int = numHeads/numHeadsKV
//let maxSeqLen = 128  // Example maximum sequence length
let maxSeqLen = 2048
let freqsCis = createFreqsCis(headDim: headDim, maxSeqLen: maxSeqLen)

//modelRunTests()

//modelProfile()


let goCapture = false
var numLayers = 5
var numExperts = 2
modelData.preload(numLayers: numLayers, numExperts: numExperts)

var numTokens = 100

if goCapture {
    numLayers = 4
    numTokens = 3
}

gpu.eval()

var ticStartTime = Date()
var countToc = 0
var silent = true

func tic() {
    ticStartTime = Date()
    countToc = 0
}

func toc(_ _msg: String = "") {
    let msg = _msg=="" ? "" : "-- \(_msg)"
    if !silent {
      // print("toc \(countToc): \(Date().timeIntervalSince(ticStartTime)*1000, precision: 2) ms \(msg)")
    }
    countToc += 1
    ticStartTime = Date()
}



func runNetwork(isTest: Bool, tokens _tokens: [VectorFloat]) -> Archive{
    var tokens = _tokens
    var xkLayerTokenHead = Array(repeating: [[VectorFloat]](), count: numLayers + 1)
    var xvLayerToken = Array(repeating: [VectorFloat](), count: numLayers)
    
    var h : VectorFloat = tokens[0]

    let hiddenSize = 14336//11008
    let stateSize = 4096
    
    let x1 = VectorFloat(shape:[hiddenSize])
    let x3 = VectorFloat(shape:[hiddenSize])
    let x2 = VectorFloat(shape:[hiddenSize])
    let ffnOut = [VectorFloat]([VectorFloat(shape:[stateSize]), VectorFloat(shape:[stateSize])])
    
    let archive = Archive()

    var startTime = Date()
    tic()
//    gpu.warnOfEvals = !silent
    for thisToken in 0...numTokens {
        h = tokens[thisToken].copy()
        for layerNo in 0..<numLayers {
            let layer = modelData.layers[layerNo]!
            let h_norm = h.rmsNormed()
            h_norm.mul(by:layer.attnNorm)
            toc()
            let xq = basicMul(v: h_norm, by: layer.wq.core)
            let xk = basicMul(v: h_norm, by: layer.wk.core).repeated(kvRepeats)
            let xv = basicMul(v: h_norm, by: layer.wv.core).repeated(kvRepeats)
            let xq_heads = xq.reshaped(newCols: headDim)
            let xk_heads = xk.reshaped(newCols: headDim)
            
            for i in 0..<numHeads {
                xq_heads[i].mul(complexArray: freqsCis[thisToken])
                xk_heads[i].mul(complexArray: freqsCis[thisToken])
            }

            
            xkLayerTokenHead[layerNo].append(xk_heads)
            xvLayerToken[layerNo].append(xv)

            let xkTokenHeads = xkLayerTokenHead[layerNo]
            let xvToken = xvLayerToken[layerNo]

            let scores = calcScores(xq_heads: xq_heads, xkTokenHeads: xkTokenHeads)

            for headNo in 0..<numHeads {
                scores[headNo].softmax()
            }
            let attnOutput = sumScores(numHeads: numHeads, headDim:headDim, scores: scores, xvToken: xvToken)

            let attnFfnOut = basicMul(v: attnOutput, by: layer.wo.core)
            
            h.add(by: attnFfnOut)

            let fxn = h.rmsNormed()

            fxn.mul(by:layer.ffnNorm)
            let gateOut = VectorFloat(shape: [8])
            basicMul(v:fxn, by:layer.ffnGate, out:gateOut)
            let gateIdxs = VectorFloat(shape:[2])
            let gateVals = VectorFloat(shape:[2])
            mpsTopK(v: gateOut, topK: 2, outIndexes: gateIdxs, outValues: gateVals)
            toc("attn")

            gpu.eval()
            toc("eval attn")
            if !silent {
//                gpu.startCapture()
//                gpu.eval()
            }

            let experts = [ExpertFfn]([layer.experts[Int(gateIdxs.getInt(index: 0))%numExperts],
                                       layer.experts[Int(gateIdxs.getInt(index: 1))%numExperts]
                                      ])
            gateVals.softmax()
            for i in 0..<2 {
                let expert = experts[i]
                bucketMul(v: fxn, by: expert.w1, out: x1, quant: 0.5)
                bucketMul(v: fxn, by: expert.w3, out: x3, quant: 0.5)
                silu(x1, x3, out: x2)
                bucketMul(v: x2, by: expert.w2, out: ffnOut[i], quant: 0.5)
                ffnOut[i].mul(by: gateVals.scalarAt(i))
            }

            h.add(by: ffnOut[0])
            h.add(by: ffnOut[1])
            toc("ffn")
//            gpu.eval()
            toc("final eval")
            gpu.stopCapture()

        }
        if silent {
            return archive
        }

        archive["token \(thisToken)"] = h.copy()
        let outNormed = h.rmsNormed()
        outNormed.mul(by: modelData.norm.asVector())

        let outputVector = VectorFloat(shape:[modelData.output.outSize])
        basicMul(v: outNormed, by: modelData.output.core, out: outputVector)
        let topKVector = mpsTopK(v: outputVector)
        print("Token \(thisToken), prep time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
        startTime = Date()
        gpu.eval()
        print("Token \(thisToken), eval time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")

        let topToken = Int(topKVector.getInt(index: 0))
        print("Token : ", t[topToken])

        if tokens.count-1 == thisToken {
          //  gpu.eval()
          //  let topToken = Int(topKVector.getInt(index: 0))
            let tokEmbeddings = modelData.tokEmbeddings.asVectorList()
            tokens.append(tokEmbeddings[topToken].asFloat32())
        }
    }

    let evalTime = Date()
    gpu.eval()
    
    print("final eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")
    print("avg time per token \(Date().timeIntervalSince(evalTime)*1000/7,  precision: 2)")
    print("tok per sec \(1000/(Date().timeIntervalSince(evalTime)*1000/7),  precision: 2)")

    print("total time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
    
    return archive
}


var errors = [String: Int]()
let i = 0
print("##### iteration", i)
let a1 = runNetwork(isTest: true, tokens: tokens)
print("##### iteration", 2)
silent = false

let evalTime = Date()
let a2 = runNetwork(isTest: true, tokens: tokens)
print("final eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")

exit(0)
/*
print(tokens.count)
let a2 = runNetwork(isTest: true, tokens: tokens)

var sumSim : Float = 0.0
for (key, _) in a1 {
//    print(key, a1[key].str())
//    print(key, a2[key].str())
    let sim = a1[key].cosineSimilarityTo(a2[key])[0]
    if key != "token 0" {
        print(key, sim, sumSim)
        sumSim += sim
    }
}
print(sumSim/9)

if sumSim > 0.85 {
    print("✅ works")
} else {
    fatalError("❌ bad quality")
}
 
 
 /*
 if (!isTest) {
     for i in 0..<2 {
         let expert = experts[i]
         mpsMul(v: fxn, by:expert.w1, out: x1)
         mpsMul(v: fxn, by:expert.w3, out: x3)
         silu(x1, x3, out: x2)
         mpsMul(v: x2, by: expert.w2, out: ffnOut[i])
         ffnOut[i].mul(by: gateVals.scalarAt(i))
     }
 } else {*/
*/
