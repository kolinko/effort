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


let headDim = 128  // Example head dimension
let numHeads = 32
let maxSeqLen = 128  // Example maximum sequence length
let freqsCis = createFreqsCis(headDim: headDim, maxSeqLen: maxSeqLen)

//modelRunTests()

//modelProfile()
//exit(0)

let goCapture = true
var numLayers = 32
var numTokens = 8

if goCapture {
    numLayers = 4
    numTokens = 3
}

//var xkLayerTokenHead = Array(repeating: [[Vector]](), count: numLayers + 1)
var xvLayerToken = Array(repeating: [Vector](), count: numLayers + 1)

gpu.eval()

func runNetwork(isTest: Bool) -> Archive{
    
    var xkLayerTokenHead = Array(repeating: [[Vector]](), count: numLayers + 1)
    
    xvLayerToken = Array(repeating: [Vector](), count: numLayers)
    /*
    gpu.eval()
     */
    
    var h : Vector = tokens[0]

    let hiddenSize = 11008
    let stateSize = 4096

    let x1 = Vector(shape:[hiddenSize])
    let x3 = Vector(shape:[hiddenSize])
    let x2 = Vector(shape:[hiddenSize])
    let ffn_out = Vector(shape:[stateSize])

    let x1_32 = VectorFloat(shape:[hiddenSize])
    let x3_32 = VectorFloat(shape:[hiddenSize])
    let x2_32 = VectorFloat(shape:[hiddenSize])
    let ffn_out32 = VectorFloat(shape:[stateSize])

    let archive = Archive()

    print("Begin token calc")
    var startTime = Date()
    for thisToken in 0..<numTokens {
        h = tokens[thisToken].copy()

        for layerNo in 0..<numLayers {
            archive.addPrefix = "\(thisToken):\(layerNo):a:"
            archive.addIdx = 0
            let layer = modelData.layers[layerNo]!
            archive.add(h)
            let h_norm = h.rmsNormed()
            archive.add(h_norm)
            h_norm.mul(byVec:layer.attnNorm)
            archive.add(h_norm)

            let xq = mpsMul(v: h_norm, by: layer.wq)
            let xk = mpsMul(v: h_norm, by: layer.wk)
            let xv = mpsMul(v: h_norm, by: layer.wv)
            archive.add([xq, xk, xv])

            let xq_heads = xq.reshaped(newCols: headDim)
            let xk_heads = xk.reshaped(newCols: headDim)
            archive.addPrefix = "\(thisToken):\(layerNo):b:"

            for i in 0..<numHeads {
                xq_heads[i].mul(complexArray: freqsCis[thisToken])
                xk_heads[i].mul(complexArray: freqsCis[thisToken])
            }
            
            xkLayerTokenHead[layerNo].append(xk_heads)
            xvLayerToken[layerNo].append(xv)
            archive.addPrefix = "\(thisToken):\(layerNo):b':"

            archive.add(xk_heads)
            archive.addPrefix = "\(thisToken):\(layerNo):b'':"

            archive.add(xv)

            let xkTokenHeads = xkLayerTokenHead[layerNo]
            let xvToken = xvLayerToken[layerNo]
            archive.addPrefix = "\(thisToken):\(layerNo):b1:"
            archive.add(xq_heads)
            archive.addPrefix = "\(thisToken):\(layerNo):b2:"

            for i in xkTokenHeads {
                archive.add(i)
            }
            archive.addPrefix = "\(thisToken):\(layerNo):b3:"

            let scores = calcScores(xq_heads: xq_heads, xkTokenHeads: xkTokenHeads)
            archive.add(scores)
            archive.addPrefix = "\(thisToken):\(layerNo):c:"

            for headNo in 0..<numHeads {
                scores[headNo].softmax()
            }
            archive.add(scores)
            archive.addPrefix = "\(thisToken):\(layerNo):c':"

            let attnOutput = sumScores(numHeads: numHeads, headDim:headDim, scores: scores, xvToken: xvToken)
            archive.add(attnOutput)

            let attnFfnOut = mpsMul(v: attnOutput, by: layer.wo)
            archive.add(attnFfnOut)
            archive.addPrefix = "\(thisToken):\(layerNo):d:"

            h.add(by: attnFfnOut)
            archive.add(h)
            archive.addPrefix = "\(thisToken):\(layerNo):e:"

            let fxn = h.rmsNormed()
            archive.add(fxn)
            archive.addPrefix = "\(thisToken):\(layerNo):f:"

            fxn.mul(byVec:layer.ffnNorm)
            archive.add(fxn)
            archive.addPrefix = "\(thisToken):\(layerNo):g:"

            archive.addPrefix = "\(thisToken):\(layerNo):=====:"
            archive.add(fxn, seriously: true)

            if isTest {
                x1_32.zero()
                x3_32.zero()
                ffn_out32.zero()
                bucketMul(v: fxn, by:layer.w1, out: x1_32, quant:0.15)
                bucketMul(v: fxn, by:layer.w3, out: x3_32, quant:0.15)
                silu(x1_32, x3_32, out: x2_32)
                bucketMul(v: x2_32, by: layer.w2, out: ffn_out32, quant: 0.15)
                ffn_out.copyFrom32(ffn_out32)
            } else {
                mpsMul(v: fxn, by:layer.w1, out: x1)
                mpsMul(v: fxn, by:layer.w3, out: x3)
                silu(x1, x3, out: x2)
                mpsMul(v: x2, by: layer.w2, out: ffn_out)
            }
            archive.addPrefix = "\(thisToken):\(layerNo):-----:"
//            archive.add([x1.asFloat16Vector(), x2, x3, ffn_out], seriously: true)
            h.add(by: ffn_out)
            archive.addPrefix = "\(thisToken):\(layerNo):h:"

            //archive.add([h], seriously: true)
        }

        archive["token \(thisToken)"] = h.copy()
        
        print("Token \(thisToken), prep time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
        if (thisToken == 0) {
            //let evalTime = Date()
            if goCapture {
                gpu.startCapture()
            }
            gpu.eval()
            //print("eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")
            
            startTime = Date()

        }
        
    }

    let evalTime = Date()
    gpu.eval()
    
    print("final eval time \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")
    print("avg time per token \(Date().timeIntervalSince(evalTime)*1000/7,  precision: 2)")
    print("tok per sec \(1000/(Date().timeIntervalSince(evalTime)*1000/7),  precision: 2)")

    print("total time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")

    if ((numTokens == 8) ){
            print(h.str())
        //    assert(h.test(mul: 10, val: [666, 666, -6.4, -1.7, -4.7, -3.0, 2.5, 2.7, -3.8, -4.70]))
            print("output OK")
        } else if (!goCapture){
            print("WARNING: Wrong token number, considering no gpucapture: \(numTokens)")
        }
    
    return archive
}


var errors = [String: Int]()
let i = 0
print("##### iteration", i)
let a1 = runNetwork(isTest: false)
let a2 = runNetwork(isTest: true)

for (key, _) in a1 {
    print(key, a1[key].str())
    print(key, a2[key].str())
    print(key, a1[key].cosineSimilarityTo(a2[key])[0])
}


    
print("done")
gpu.stopCapture()
