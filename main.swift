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
print("loading")

os_signpost(.begin, log: log, name: "Loading")
let modelData = loadModelData(from: "shape.json")
let tokens = loadTokens()
os_signpost(.end, log: log, name: "Loading")

//let dim = 4096
//let dim_range = 0...4095

let headDim = 128  // Example head dimension
let numHeads = 32
let maxSeqLen = 128  // Example maximum sequence length
let freqsCis = createFreqsCis(headDim: headDim, maxSeqLen: maxSeqLen)

//let tokenNum = 0

let goCapture = false
var numLayers = 32
var numTokens = 8

if goCapture {
    numLayers = 4
    numTokens = 3
}

var xkLayerTokenHead = Array(repeating: [[Vector]](), count: numLayers + 1)
var xqLayerTokenHead = Array(repeating: [[Vector]](), count: numLayers + 1)
var xvLayerToken = Array(repeating: [Vector](), count: numLayers + 1)

modelRunTests()

modelProfile()

print("tokenCalc")
var h : Vector = tokens[0]

let hiddenSize = 11008
let stateSize = 4096

let ffn_out = Vector(shape:[stateSize])
let x1 = Vector(shape:[hiddenSize])
let x3 = Vector(shape:[hiddenSize])
let x2 = Vector(shape:[hiddenSize])
let attnFfnOut = Vector(shape:[modelData.layers[0]!.wo.outSize])

var startTime = Date()
for thisToken in 0..<numTokens {
    h = tokens[thisToken]

    for layerNo in 0..<numLayers {
        let layer = modelData.layers[layerNo]!
        
        let h_norm = h.rmsNormed()
        h_norm.mul(byVec:layer.attnNorm)
        
        let xq = mpsMul(v: h_norm, by: layer.wq)
        let xk = mpsMul(v: h_norm, by: layer.wk)
        let xv = mpsMul(v: h_norm, by: layer.wv)

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

        mpsMul(v: attnOutput, by: layer.wo, out: attnFfnOut)
        h.add(by: attnFfnOut)
        
        let fxn = h.rmsNormed()

        fxn.mul(byVec:layer.ffnNorm)

        mpsMul(v: fxn, by:layer.w1, out: x1)
        mpsMul(v: fxn, by:layer.w3, out: x3)
        silu(x1, x3, out: x2)
        mpsMul(v: x2, by: layer.w2, out: ffn_out)
        h.add(by: ffn_out)
    }
    
    print("Token \(thisToken), prep time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
    if (thisToken == 0) {
        let evalTime = Date()
        if goCapture {
            gpu.startCapture()
        }
        gpu.eval()
        print("eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")
        
        startTime = Date()

    }
    
}
//exit(0)
let evalTime = Date()
gpu.eval()



print("final eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")



print("avg time per token \(Date().timeIntervalSince(evalTime)*1000/7,  precision: 2)")
print("tok per sec \(1000/(Date().timeIntervalSince(evalTime)*1000/7),  precision: 2)")

print("total time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")

if (numTokens == 8) {
    print(h.str())
    assert(h.test(mul: 10, val: [666, 666, -6.4, -1.7, -4.7, -3.0, 2.5, 2.7, -3.8, -4.70]))
    print("output OK")
} else if (!goCapture){
    print("WARNING: Wrong token number, considering no gpucapture")
}
print("done")
gpu.stopCapture()

exit(0)
