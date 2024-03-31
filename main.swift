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

let goCapture = false
var numLayers = 2
var numExperts = 2

var numTokens = 100

if goCapture {
    numLayers = 4
    numTokens = 3
}

let bam = BufferActivityManager()
bam.startPeriodicDispatch()
let modelData = Model(from: "shape.json", numLayers: numLayers, numExperts: numExperts, percentLoad: 0x0C)

var tokens = [VectorFloat]()
let tokIds = [1, 1602, 460] // "How are"
/*
let tokIds = [1,
              523,
              28713,
              28767,
              28792,
              16289,
              28793,
              26703,
              349,
              6084,
              387,
              14469,
              442,
              6444,
              300,
              28804,
              28792,
              28748,
              16289,
              28793
]*/
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
let maxTokens = 2048
let hiddenSize = 14336//11008
let stateSize = 4096
let freqsCis = createFreqsCis(headDim: headDim, maxSeqLen: maxSeqLen)

//modelRunTests()

//modelProfile()


print()
gpu.eval()

var silent = true


func runNetwork(isTest: Bool, tokens _tokens: [VectorFloat], quant: Double = 1.0) -> Archive{

    var tokens = _tokens
    let xkLayerTokenHead = Matrix4DFloat(shape:[numLayers, maxTokens, numHeads, headDim])
    let xvLayerToken = Matrix3DFloat(shape:[numLayers, maxTokens, stateSize])
    
    var h : VectorFloat = tokens[0]

    let x1 = VectorFloat(shape:[hiddenSize])
    let x3 = VectorFloat(shape:[hiddenSize])
    let x2 = VectorFloat(shape:[hiddenSize])
    let ffnOut = [VectorFloat]([VectorFloat(shape:[stateSize]), VectorFloat(shape:[stateSize])])
    
    let archive = Archive()

    var output = ""
    
    gpu.warnOfEvals = false
    let finalEvalTime = Date()
    var evalTime = Date()
    var sumPrepTime = Date().timeIntervalSince(evalTime)
    var sumEvalTime = Date().timeIntervalSince(evalTime)
    for thisToken in 0...numTokens {
/*        if thisToken == 2 {
            gpu.eval()
            gpu.startCapture()
            gpu.eval()
        }*/

        h = tokens[thisToken].copy()
        for layerNo in 0..<numLayers {
            let layer = modelData.layers[layerNo]!
            let h_norm = h.rmsNormed()
            h_norm.mul(by:layer.attnNorm)
            let xq = basicMul(v: h_norm, by: layer.wq.core)
            let xk = basicMul(v: h_norm, by: layer.wk.core).repeated(kvRepeats)
            let xqHeads = xq.asMatrix(newCols: headDim)
            let xkHeads = xk.asMatrix(newCols: headDim)
            
            for i in 0..<numHeads {
                xqHeads[i].mul(complexArray: freqsCis[thisToken])
                xkHeads[i].mul(complexArray: freqsCis[thisToken])
            }
            
            xkLayerTokenHead[layerNo][thisToken].copyFrom(xkHeads)
            
            basicMul(v: h_norm, by: layer.wv.core).repeated(kvRepeats, into: xvLayerToken[layerNo][thisToken])

            let xkTokenHeads = xkLayerTokenHead[layerNo]
            let xvToken = xvLayerToken[layerNo]
            
            let scores = calcScores2(xq_heads: xqHeads, xkTokenHeads: xkTokenHeads, numTokens: thisToken+1)
            
            if numExperts == 2 && numLayers == 2 {
                gpu.eval()
                
                if thisToken == 0 && layerNo == 0 { gpu.eval(); assert (Int(scores.asVectorList()[0][0]*10000) == 1021)}
                if thisToken == 1 && layerNo == 0 {
                    gpu.eval()
                    gpu.stopCapture()
                    assert(Int(scores.asVectorList()[17][1]*10000) == -24692)
                }
                if thisToken == 1 && layerNo == 1 {
                    print("hello")
                }
            }
                
            for headNo in 0..<numHeads {
                scores.asVectorList()[headNo].softmax()
            }
            let attnOutput = sumScores2(numHeads: numHeads, headDim:headDim, scores: scores, xvToken: xvToken, numTokens: thisToken+1)

            let attnFfnOut = basicMul(v: attnOutput, by: layer.wo.core)
            
            h.add(by: attnFfnOut)

            let fxn = h.rmsNormed()

            fxn.mul(by:layer.ffnNorm)
            let gateOut = VectorFloat(shape: [numExperts])
            basicMul(v:fxn, by:layer.ffnGate, out:gateOut)
            let gateIdxs = VectorFloat(shape:[2])
            let gateVals = VectorFloat(shape:[2])
            mpsTopK(v: gateOut, topK: 2, outIndexes: gateIdxs, outValues: gateVals)
            if !silent {
//                gpu.startCapture()
//                gpu.eval()
            }

            gateVals.softmax()
            for i in 0..<2 {
                let expIdx = gateIdxs.scalarAt(i)
                expertMul(v: fxn, by: layer.w1, expNo: expIdx, out: x1, quant: quant)
                expertMul(v: fxn, by: layer.w3, expNo: expIdx, out: x3, quant: quant)

                silu(x1, x3, out: x2)
                expertMul(v: x2, by: layer.w2, expNo: expIdx, out: ffnOut[i], quant: quant)
                ffnOut[i].mul(by: gateVals.scalarAt(i))
            }

            h.add(by: ffnOut[0])
            h.add(by: ffnOut[1])

        }

        archive["token \(thisToken)"] = h.copy()
        let outNormed = h.rmsNormed()
        outNormed.mul(by: modelData.norm.asVector())

        let outputVector = VectorFloat(shape:[modelData.output.outSize])
        basicMul(v: outNormed, by: modelData.output.core, out: outputVector)
        let topKVector = mpsTopK(v: outputVector)

        sumPrepTime += Date().timeIntervalSince(evalTime)
        let ptime = Date().timeIntervalSince(evalTime)*1000
        evalTime = Date()
        gpu.eval()
       // print(thisToken, topKVector.strInt)
        /*
        if thisToken < 2 {
            assert(topKVector.getInt(index: 0) == [18816, 31739][thisToken])//, 3971, 25215, 2810, 20686, 9608, 20686, 9608, 20686, 9608, 20686][thisToken])
        }*/
            
        if !silent {
            sumEvalTime += Date().timeIntervalSince(evalTime)
            print("prep: \(ptime, precision: 2) ms; eval: \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")
            evalTime = Date()
        }

        let topToken = Int(topKVector.getInt(index: 0))

        if tokens.count-1 == thisToken {
            let tokEmbeddings = modelData.tokEmbeddings.asVectorList()
            tokens.append(tokEmbeddings[topToken].asFloat32())
            output += t[topToken].replacingOccurrences(of: "▁", with: " ")

        }
    }

    evalTime = Date()
    gpu.eval()
    gpu.stopCapture()

    if let range = output.range(of: "</s>") {
        output = String(output.prefix(upTo: range.lowerBound)) + "ₔ"
    } else {
        output += " ›››"
    }
    
    if !silent {
        print("\(Int(quant*100))% \t \(output)\n")
        
        print("final eval time \(Date().timeIntervalSince(finalEvalTime)*1000, precision: 2) ms")
        
        print("sum eval time \(sumEvalTime*1000, precision: 2) ms")
        print("sum prep time \(sumPrepTime*1000, precision: 2) ms")
        print("avg eval time \(sumEvalTime*1000/Double(numTokens), precision: 2) ms")
        print("avg prep time \(sumPrepTime*1000/Double(numTokens), precision: 2) ms")
        
        print("total \(1000/((sumEvalTime+sumEvalTime)*1000/Double(numTokens)), precision: 2) tps")
    }

    return archive
}

var runControl = false
silent = false
//_ = control(isTest: true, tokens: tokens, quant:0.30)
_ = runNetwork(isTest: true, tokens: tokens, quant:0.30)

runControl = false
_ = runNetwork(isTest: true, tokens: tokens, quant:0.30)

/*
for i in 2...25 {
    let _ = runNetwork(isTest: true, tokens: tokens, quant:Double(i*2)/100)
}*/
