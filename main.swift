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

let captureGPU = false

let log = OSLog(subsystem: "com.kolinko", category: "Performance")
 
let gpu = Gpu()
print("loading")
os_signpost(.begin, log: log, name: "Loading")

let modelData = loadModelData(from: "shape.json")
let tokens = loadTokens()
os_signpost(.end, log: log, name: "Loading")

let dim = 4096
let dim_range = 0...4095

let headDim = 128  // Example head dimension
let numHeads = 32
let maxSeqLen = 128  // Example maximum sequence length
let freqsCis = createFreqsCis(headDim: headDim, maxSeqLen: maxSeqLen)

let tokenNum = 0

let numLayers = 31
let numTokens = 8

var xkLayerTokenHead = Array(repeating: [[Vector]](), count: numLayers + 1)
var xqLayerTokenHead = Array(repeating: [[Vector]](), count: numLayers + 1)
var xvLayerToken = Array(repeating: [Vector](), count: numLayers + 1)

os_signpost(.begin, log: log, name: "Go Tokens3")

let startTime = Date()

import Foundation
import simd

gpu.startCapture(cond: captureGPU)

for thisToken in 0..<numTokens {
    for layerNo in 0...numLayers {
        var h = tokens[thisToken]
        let layer = modelData.layers[layerNo]!
        
        let wa = layer["attention_norm"]!.asVector()
        
        let wq = layer["attention.wq"]!
        let wk = layer["attention.wk"]!
        let wv = layer["attention.wv"]!
        
        let wo = layer["attention.wo"]!
        
        let h_norm = h.rmsNormed()
        h_norm.mul(by:wa)
        
        let xq = mul_col(vec: h_norm, by: wq)
        let xk = mul_col(vec: h_norm, by: wk)
        let xv = mul_col(vec: h_norm, by: wv)
        let xq_heads = xq.reshaped(newCols: headDim)
        let xk_heads = xk.reshaped(newCols: headDim)
        
        for i in 0..<numHeads {
            xq_heads[i].mul(complexArray: freqsCis[tokenNum])
            xk_heads[i].mul(complexArray: freqsCis[tokenNum])
        }
        
        xkLayerTokenHead[layerNo].append(xk_heads)
        xvLayerToken[layerNo].append(xv)
        
        let xkTokenHeads = xkLayerTokenHead[layerNo]
        let xvToken = xvLayerToken[layerNo]

        var scores = calcScores(xq_heads: xq_heads, xkTokenHeads: xkTokenHeads)
        for headNo in 0..<numHeads {
            softmax(&scores[headNo])
        }
        assert(xvToken[0].test("attnFfn", cond: layerNo+thisToken==0, mul:1000, val:[-0.001, 0.006, -0.006, 0.028, -0.028]))
        
        let outMatrix = sumScores(numHeads: numHeads, headDim:headDim, scores: scores, xvToken: xvToken)

        let attnOutput = outMatrix.asVector()

        let attnFfn = mul_col(vec: attnOutput, by: wo)
        assert(attnFfn.test("attnFfn", cond: layerNo+thisToken==0, mul:100, val:[-0.05, -0.02, -0.09, -0.07, -0.04]))
        
                
        h.add(by: attnFfn)
        assert(h.test("h", cond: layerNo+thisToken==0, mul:100, val:[-0.03, -0.03, -0.07, -0.04, -0.05]))

        let fxn = h.rmsNormed()
        assert(fxn.test("h_norm2", cond: layerNo+thisToken==0, mul:100, val:[-0.74, -0.69, -1.71, -0.949, -1.246]))
        
        let wn = layer["ffn_norm"]!.asVector()
        let w1 = layer["feed_forward.w1"]!
        let w2 = layer["feed_forward.w2"]!
        let w3 = layer["feed_forward.w3"]!
        
        fxn.mul(by:wn)
        ffn(&h, fxn:fxn, w1:w1, w2:w2, w3:w3)
        
        assert(fxn.test("fxn", cond: layerNo+thisToken==0, mul:100, val:[-0.04, -0.06, -0.14, -0.07, -0.09]))
        assert(h.test("h", cond: layerNo+thisToken==0, mul:100, val:[-0.06,-0.12,-0.05,-0.09,0.01,-0.01,-0.07]))
    }
    
    print("Token \(thisToken), prep time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
    if (thisToken == 0) {
        let evalTime = Date()

        gpu.eval()

        print("eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")

        gpu.stopCapture(cond: captureGPU)        

    }
    
}
os_signpost(.end, log: log, name: "Go Tokens3")

let evalTime = Date()
os_signpost(.begin, log: log, name: "Go Eval")
gpu.eval()
os_signpost(.end, log: log, name: "Go Eval")
print("final eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")


print("avg time per token \(Date().timeIntervalSince(evalTime)*1000/7,  precision: 2)")
print("tok per sec \(1000/(Date().timeIntervalSince(evalTime)*1000/7),  precision: 2)")

print("total time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
print("done")
exit(0)
/*

*/
