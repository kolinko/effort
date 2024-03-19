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

//let dim = 4096
//let dim_range = 0...4095

let headDim = 128  // Example head dimension
let numHeads = 32
let maxSeqLen = 128  // Example maximum sequence length
let freqsCis = createFreqsCis(headDim: headDim, maxSeqLen: maxSeqLen)

let tokenNum = 0

let numLayers = 5
let numTokens = 8

var xkLayerTokenHead = Array(repeating: [[Vector]](), count: numLayers + 1)
var xqLayerTokenHead = Array(repeating: [[Vector]](), count: numLayers + 1)
var xvLayerToken = Array(repeating: [Vector](), count: numLayers + 1)

os_signpost(.begin, log: log, name: "Go Tokens3")

var startTime = Date()

import Foundation
import simd

modelRunTests()

//gpu.startCapture(cond: captureGPU)
/*
let captureManager = MTLCaptureManager.shared()
let captureDescriptor = MTLCaptureDescriptor()
captureDescriptor.captureObject = gpu.device
do {
    try captureManager.startCapture(with: captureDescriptor)
} catch {
    fatalError("error when trying to capture: \(error)")
}*/

print("begin")
let layer = modelData.layers[0]!
let rowVals = layer["feed_forward.w1"+".vals"]!
let seedVec = VectorFloat(shape:[rowVals.cols!])
let outDim: Int =  Int(rowVals.cols!) / 16


var h = tokens[0]
let bufferX = VectorFloat(shape:[rowVals.cols!])
let buffer16 = Vector(shape:[rowVals.cols!])


for _ in 0..<2 {
    for layerNo in 0..<32 {
        let layer = modelData.layers[layerNo]!
        let rowVals = layer["feed_forward.w1"+".vals"]!
        //let bufferX = VectorFloat(shape:[rowVals.cols!])
        //let outDim: Int =  Int(rowVals.cols!) / 16
        mpsMul(vector: h, weights: layer["feed_forward.w1"]!, result: buffer16)

        gpu.deploy("bucketMul", buffers: [h, rowVals, bufferX], ints:[rowVals.rows, outDim], threadCount: outDim * 32) // 32 =
    }
}
gpu.eval()
print("initial time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")

gpu.startCapture(cond: captureGPU)
gpu.eval()
h = tokens[0]

print("begin benchmark")
startTime = Date()
for i in 0..<rowVals.cols! {
    seedVec[i] = Float.random(in:0..<1)
}
let repeats = 100
for _ in 0..<repeats*4 {
    //    h = tokens[thisToken]
    for layerNo in 0..<32 {
        let layer = modelData.layers[layerNo]!
        let rowVals = layer["feed_forward.w1"+".vals"]!
//        mpsMul(vector: h, weights: layer["feed_forward.w1"]!, result: buffer16)
        gpu.deploy("bucketMul", buffers: [h, rowVals, bufferX], ints:[rowVals.rows, outDim], threadCount: outDim * 32)
    }
}
print("prep time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
startTime = Date()
gpu.eval()
print("final time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms, \(Date().timeIntervalSince(startTime)*1000/Double(repeats), precision: 2)ms")
print(bufferX[0])
print("done")
gpu.stopCapture(cond: captureGPU)
exit(0)
for thisToken in 0..<numTokens {
    var h = tokens[thisToken]

    for layerNo in 0...numLayers {
//        print("layer", layerNo, thisToken)
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
        let scores = calcScores(xq_heads: xq_heads, xkTokenHeads: xkTokenHeads)

        for headNo in 0..<numHeads {
            scores[headNo].softmax()
        }
//        assert(xvToken[0].test("attnFfn", cond: layerNo+thisToken==0, mul:1000, val:[-0.001, 0.006, -0.006, 0.028, -0.028]))
        
        let attnOutput = sumScores(numHeads: numHeads, headDim:headDim, scores: scores, xvToken: xvToken)

        let attnFfn = mul_col(vec: attnOutput, by: wo)

//        assert(attnFfn.test("attnFfn", cond: layerNo+thisToken==0, mul:100, val:[-0.05, -0.02, -0.09, -0.07, -0.04]))

        h.add(by: attnFfn)
//        assert(h.test("h", cond: layerNo+thisToken==0, mul:100, val:[-0.03, -0.03, -0.07, -0.04, -0.05]))

        let fxn = h.rmsNormed()
//        assert(fxn.test("h_norm2", cond: layerNo+thisToken==0, mul:100, val:[-0.74, -0.69, -1.71, -0.949, -1.246]))
        
        let wn = layer["ffn_norm"]!.asVector()
        let w1 = layer["feed_forward.w1"]!
        let w2 = layer["feed_forward.w2"]!
        let w3 = layer["feed_forward.w3"]!
        
        fxn.mul(by:wn)
        //threadExecutionWidth
        
        let x1 = mul_vm(v: fxn, layer: layer, name: "feed_forward.w1")
        let x3 = mul_vm(v: fxn, layer: layer, name: "feed_forward.w3")
        let x2 = silu(x1, x3)
        let ffn_out = mul_vm(v: x2, layer: layer, name: "feed_forward.w2")
        h.add(by: ffn_out.asFloat16Vector())
        
        ffn(&h, fxn:fxn, w1:w1, w2:w2, w3:w3)
        /*

        if (thisToken == 1 && layerNo == 10) {
            gpu.eval()
        }
        if (thisToken == 1 && layerNo == 11) {
            gpu.eval()
//            gpu.stopCapture()
        }*/

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
//captureManager.stopCapture()

print("final eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")


print("avg time per token \(Date().timeIntervalSince(evalTime)*1000/7,  precision: 2)")
print("tok per sec \(1000/(Date().timeIntervalSince(evalTime)*1000/7),  precision: 2)")

print("total time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
print("done")
exit(0)
