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
var numLayers = 32
var numExperts = 8


var numTokens = 100

if goCapture {
    numLayers = 4
    numTokens = 3
}

let bam = BufferActivityManager()
bam.startPeriodicDispatch()
let modelData = Model(from: "shape.json", numLayers: numLayers, numExperts: numExperts, percentLoad: 0x0C)

var tokens = [VectorFloat]()
//let tokIds = [1, 1602, 460] // "How are"
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
]
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


print()
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



func runNetwork(isTest: Bool, tokens _tokens: [VectorFloat], quant: Double = 1.0) -> Archive{
    let origCount = _tokens.count
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

    var output = ""
    
    tic()
    gpu.warnOfEvals = false
    var evalTime = Date()
    var finalEvalTime = Date()

    var sumPrepTime = Date().timeIntervalSince(evalTime)
    var sumEvalTime = Date().timeIntervalSince(evalTime)
    
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
            let gateOut = VectorFloat(shape: [numExperts])
            basicMul(v:fxn, by:layer.ffnGate, out:gateOut)
            let gateIdxs = VectorFloat(shape:[2])
            let gateVals = VectorFloat(shape:[2])
            mpsTopK(v: gateOut, topK: 2, outIndexes: gateIdxs, outValues: gateVals)
            toc("attn")

            toc("eval attn")
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
            toc("ffn")
            //if thisToken % 20 == 0 {
            //    gpu.eval()
            //}
            toc("final eval")
            gpu.stopCapture()

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
        
        sumEvalTime += Date().timeIntervalSince(evalTime)
        print("prep: \(ptime, precision: 2) ms; eval: \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")
        evalTime = Date()


        let topToken = Int(topKVector.getInt(index: 0))
        

        if tokens.count-1 == thisToken {
            let tokEmbeddings = modelData.tokEmbeddings.asVectorList()
            tokens.append(tokEmbeddings[topToken].asFloat32())
            output += t[topToken].replacingOccurrences(of: "▁", with: " ")

        }
    }


    evalTime = Date()
    gpu.eval()
    
    if let range = output.range(of: "</s>") {
        output = String(output.prefix(upTo: range.lowerBound)) + "ₔ"
    } else {
        output += " ›››"
    }
    
    print("\(Int(quant*100))% \t \(output)\n")

    print("final eval time \(Date().timeIntervalSince(finalEvalTime)*1000, precision: 2) ms")
    
    print("sum eval time \(sumEvalTime*1000, precision: 2) ms")
    print("sum prep time \(sumPrepTime*1000, precision: 2) ms")
    print("avg eval time \(sumEvalTime*1000/Double(numTokens), precision: 2) ms")
    print("avg prep time \(sumPrepTime*1000/Double(numTokens), precision: 2) ms")
    
    print("total \(1000/((sumEvalTime+sumEvalTime)*1000/Double(numTokens)), precision: 2) tps")

    return archive
}


var errors = [String: Int]()
silent = false
for i in 2...25 {
    let _ = runNetwork(isTest: true, tokens: tokens, quant:Double(i*2)/100)
}
