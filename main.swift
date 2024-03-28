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

let modelData = Model(from: "shape.json")

var tokens = [Vector]()
let tokIds = [1, 1602, 460] // "How are"
let t = Tokeniser()

let tokEmbeddings = modelData.tokEmbeddings.asVectorList()
for t in tokIds {
    tokens.append(tokEmbeddings[t])
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
var numLayers = 32
var numTokens = 100

if goCapture {
    numLayers = 4
    numTokens = 3
}

gpu.eval()


func runNetwork(isTest: Bool, tokens _tokens: [Vector]) -> Archive{
    var tokens = _tokens
    var xkLayerTokenHead = Array(repeating: [[Vector]](), count: numLayers + 1)
    var xvLayerToken = Array(repeating: [Vector](), count: numLayers)
    
    var h : Vector = tokens[0]

    let hiddenSize = 14336//11008
    let stateSize = 4096

    let x1 = Vector(shape:[hiddenSize])
    let x3 = Vector(shape:[hiddenSize])
    let x2 = Vector(shape:[hiddenSize])
    var ffnOut = [Vector]([Vector(shape:[stateSize]), Vector(shape:[stateSize])])
    
    let x1_32 = VectorFloat(shape:[hiddenSize])
    let x3_32 = VectorFloat(shape:[hiddenSize])
    let x2_32 = VectorFloat(shape:[hiddenSize])
    let ffn_out32 = [VectorFloat]([VectorFloat(shape:[stateSize]), VectorFloat(shape:[stateSize])])
    
    let archive = Archive()

    print("Begin token calc")
    var startTime = Date()
    for thisToken in 0...numTokens {
        h = tokens[thisToken].copy()
        for layerNo in 0..<numLayers {
            let layer = modelData.layers[layerNo]!
            let h_norm = h.rmsNormed()
            h_norm.mul(by:layer.attnNorm)

            let xq = mpsMul(v: h_norm, by: layer.wq)
            let xk = mpsMul(v: h_norm, by: layer.wk).repeated2(kvRepeats)
            let xv = mpsMul(v: h_norm, by: layer.wv).repeated2(kvRepeats)
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

            let attnFfnOut = mpsMul(v: attnOutput, by: layer.wo)
            
            h.add(by: attnFfnOut)

            let fxn = h.rmsNormed()

            fxn.mul(by:layer.ffnNorm)
            let gateOut = Vector(shape: [8])
            mpsMul(v:fxn, by:layer.ffnGate, out:gateOut)
            let gateIdxs = VectorFloat(shape:[2])
            let gateVals = Vector(shape:[2])
            mpsTopK(v: gateOut, topK: 2, outIndexVector: gateIdxs, outValueVector: gateVals)
            gpu.eval()
            let experts = [ExpertFfn]([layer.experts[Int(gateIdxs.getInt(index: 0))],
                                       layer.experts[Int(gateIdxs.getInt(index: 1))]
                                      ])
            gateVals.softmax()
            if (!isTest) {
                for i in 0..<2 {
                    let expert = experts[i]
                    mpsMul(v: fxn, by:expert.w1, out: x1)
                    mpsMul(v: fxn, by:expert.w3, out: x3)
                    silu(x1, x3, out: x2)
                    mpsMul(v: x2, by: expert.w2, out: ffnOut[i])
                    ffnOut[i].mul(by: gateVals.scalarAt(i))
                }


            } else {
                for i in 0..<2 {
                    let expert = experts[i]
                    bucketMul(v: fxn, by: expert.w1, out: x1_32, quant: 0.2)
                    bucketMul(v: fxn, by: expert.w3, out: x3_32, quant: 0.2)
                    silu(x1_32, x3_32, out: x2_32)
                    bucketMul(v: x2_32, by: expert.w2, out: ffn_out32[i], quant: 0.7)
                    ffn_out32[i].mul(by: gateVals.scalarAt(i))
                    ffnOut[i] = ffn_out32[i].asFloat16Vector()
//                    ffnOut[i].mul(by: gateVals.scalarAt(i))
                }
            }

            ffnOut[0].add(by: ffnOut[1])
            h.add(by: ffnOut[0])


        }
        archive["token \(thisToken)"] = h.copy()
        let outNormed = h.rmsNormed()
        outNormed.mul(by: modelData.norm.asVector())

        let outputVector = Vector(shape:[modelData.output.outSize])
        mpsMul(v: outNormed, by: modelData.output, out: outputVector)
        let topKVector = mpsTopK(v: outputVector)
        gpu.eval()
        let topToken = Int(topKVector.getInt(index: 0))

        print("Token \(thisToken), prep time \(Date().timeIntervalSince(startTime)*1000, precision: 2) ms")
        print("Token: ", t[topToken])

        if (thisToken == 0) {
            if goCapture {
                gpu.startCapture()
            }
            print("evaling first token...")
            gpu.eval()
            startTime = Date()
        }
        
        if tokens.count-1 == thisToken {
            gpu.eval()
            let topToken = Int(topKVector.getInt(index: 0))
            let tokEmbeddings = modelData.tokEmbeddings.asVectorList()
            tokens.append(tokEmbeddings[topToken])

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
*/
/*
var s1 = t.decode(tokIds, delim: "")
var s2 = t.decode(tokIds, delim: "")

for i in 0..<numTokens {
    let key = "topK \(i)"
    print(key)
    var s = ""
    for j in 0..<a1[key].rows/2 {
        s+="\(t.decode([Int(a1[key].getInt(index: j*2))])) "
    }
    if i >= tokIds.count {
        s1+=t.decode([Int(a1[key].getInt(index: 0))], delim: "")
    }
    print(key, s)
    
    s = ""
    for j in 0..<a1[key].rows/2 {
        s+="\(t.decode([Int(a2[key].getInt(index: j*2))])) "
    }
    
    if i >= tokIds.count {
        s2+=t.decode([Int(a2[key].getInt(index: 0))], delim: "")
    }

    print(key, s)

}

print("original")
print(s1.replacingOccurrences(of: "_", with: " "))

print("approx")
print(s2.replacingOccurrences(of: "_", with: " "))

print("done")

//a1.serialize(fname: "a1")
gpu.stopCapture()
*/
