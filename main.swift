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
let devices = MTLCopyAllDevices()
assert(!devices.isEmpty, "No Metal devices available")

// Optionally, you can choose a device based on specific criteria.
// For simplicity, let's use the first available device.
let device = devices[0]
let commandQueue = device.makeCommandQueue()!

let gpu = Gpu()
print("loading")
os_signpost(.begin, log: log, name: "Loading")

let modelData = loadModelData(from: "shape.json", device: device)
let tokens = loadTokens(device: device)
os_signpost(.end, log: log, name: "Loading")
/*assert(modelData.layers[0]!["feed_forward.w1.ids"]!.testInt("w1ids", val:[3260, 7938, 9263, 9670]))*/
assert(tokens[0].test("h", mul: 100, val: [0.02, -0.01, 0.01, 0.02, -0.01]))


let library = device.makeDefaultLibrary()!
let internalSFunc = library.makeFunction(name:"internal")!
let internalSState = try! device.makeComputePipelineState(function: internalSFunc)
let secondSFunc = library.makeFunction(name:"internal")!
let secondSState = try! device.makeComputePipelineState(function: secondSFunc)
var globalStates: [String: MTLComputePipelineState] = [:]
let functionNames = ["sum_of_squares", "normalize_vector",
                     "sum_of_exps","softmax_add", "memcpy", "sumScores",
                     "dot", "setScore", "internal", "second", "mul_col_4096", "mul_vec", "add_vec", "mul_complex"] // Add more function names as needed

for fname in functionNames {
    let internalFunc = library.makeFunction(name: fname)!
    globalStates[fname] = try! device.makeComputePipelineState(function: internalFunc)
}


let dim = 4096
let dim_range = 0...4095

let headDim = 128  // Example head dimension
let numHeads = 32
let maxSeqLen = 128  // Example maximum sequence length
let freqsCis = createFreqsCis(headDim: headDim, maxSeqLen: maxSeqLen)

let tokenNum = 0


var xkLayerTokenHead = [[[Vector]]]()
var xqLayerTokenHead = [[[Vector]]]()
var xvLayerToken = [[Vector]]()

let numLayers2 = 31
for _ in 0...numLayers2 {
    xkLayerTokenHead.append([[Vector]]())
    xqLayerTokenHead.append([[Vector]]())
    xvLayerToken.append([Vector]())

}

let numTokens = 1
os_signpost(.begin, log: log, name: "Go Tokens3")

let startTime = Date()

var thisToken = 0

import Foundation
import simd

let captureManager = MTLCaptureManager.shared()
let captureDescriptor = MTLCaptureDescriptor()
if captureGPU {
    captureDescriptor.captureObject = device
    do {
        try captureManager.startCapture(with: captureDescriptor)
    } catch {
        fatalError("error when trying to capture: \(error)")
    }
}
    

for i in 0..<8 {
    thisToken = i
    for layerNo in 0...numLayers2 {
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
//        exit(0)

        
    }
    
    print("Token \(thisToken), prep time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
    if (thisToken == 0) {
        let evalTime = Date()

        gpu.eval()
        print("eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")

        print("eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")


        if (captureGPU) {
            captureManager.stopCapture()
        }
        

    }
    
}
os_signpost(.end, log: log, name: "Go Tokens3")

let evalTime = Date()
os_signpost(.begin, log: log, name: "Go Eval")
gpu.eval()
print("eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")
os_signpost(.end, log: log, name: "Go Eval")
print("final eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")


print("avg time per token \(Date().timeIntervalSince(evalTime)*1000/7,  precision: 2)")
print("tok per sec \(1000/(Date().timeIntervalSince(evalTime)*1000/7),  precision: 2)")

print("total time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
print("done")
exit(0)
/*

*/
