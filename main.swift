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

var numLayers = 32
var numExperts = 8

var numTokens = 100


let bam = BufferActivityManager()
bam.startPeriodicDispatch()
let modelData = Model(from: "shape.json", numLayers: numLayers, numExperts: numExperts, percentLoad: 0x0C)

var tokens = [VectorFloat]()
//let tokIds = [1, 1602, 460] // "How are"

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
        if thisToken == 2 {
            os_signpost(.begin, log: log, name: "TokenGen")
        }
        if thisToken == 2 {
            sumPrepTime = Date().timeIntervalSince(evalTime)
            sumEvalTime = Date().timeIntervalSince(evalTime)
        }
        
        if thisToken == 2 {
            gpu.eval()
         //   gpu.startCapture()
            gpu.eval()
        }

        h = tokens[thisToken].copy()
        for layerNo in 0..<numLayers {
            let layer = modelData.layers[layerNo]!
            let h_norm = h.rmsNormed()
            h_norm.mul(by:layer.attnNorm)
            let xq = basicMul(v: h_norm, by: layer.wq.core)
            let xk = xkLayerTokenHead[layerNo][thisToken].asVector()
            basicMul(v: h_norm, by: layer.wk.core).repeated(kvRepeats, into:xk)
            let xqHeads = xq.asMatrix(newCols: headDim)
            let xkHeads = xk.asMatrix(newCols: headDim)
            
            for i in 0..<numHeads {
                xqHeads[i].mul(complexArray: freqsCis[thisToken])
                xkHeads[i].mul(complexArray: freqsCis[thisToken])
            }
            
            //xkLayerTokenHead[layerNo][thisToken].copyFrom(xkHeads)
            
            basicMul(v: h_norm, by: layer.wv.core).repeated(kvRepeats, into: xvLayerToken[layerNo][thisToken])

            let xkTokenHeads = xkLayerTokenHead[layerNo]
            let xvToken = xvLayerToken[layerNo]
            
            let scores = calcScores2(xq_heads: xqHeads, xkTokenHeads: xkTokenHeads, numTokens: thisToken+1)
            
            if numExperts == 2 && numLayers == 2 && false{
                gpu.eval()
                
                if thisToken == 0 && layerNo == 0 { gpu.eval(); assert (Int(scores.asVectorList()[0][0]*10000) == 1022)}
                if thisToken == 1 && layerNo == 0 {
                    gpu.eval()
                    gpu.stopCapture()
                    assert(Int(scores.asVectorList()[17][1]*10000) == -24685)
                }
                if thisToken == 1 && layerNo == 1 {
                    print("hello")
                }
            }
                
            for headNo in 0..<numHeads {
                scores[headNo].softmax()
            }
            let attnOutput = sumScores2(numHeads: numHeads, headDim:headDim, scores: scores, xvToken: xvToken, numTokens: thisToken+1)
            
            let attnFfnOut = basicMul(v: attnOutput, by: layer.wo.core)
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
                if !isTest {
                    expertMul(v: fxn, by: layer.w1, expNo: expIdx, out: x1, quant: quant)
                    expertMul(v: fxn, by: layer.w3, expNo: expIdx, out: x3, quant: quant)
                    
                    silu(x1, x3, out: x2)
                    expertMul(v: x2, by: layer.w2, expNo: expIdx, out: ffnOut[i], quant: quant)
                    ffnOut[i].mul(by: gateVals.scalarAt(i))
                } else {
                    expertMul3(v: fxn, by: layer.w1, expNo: expIdx, out: x1, quant: quant)
                    expertMul3(v: fxn, by: layer.w3, expNo: expIdx, out: x3, quant: quant)
                    
                    silu(x1, x3, out: x2)
                    expertMul3(v: x2, by: layer.w2, expNo: expIdx, out: ffnOut[i], quant: quant)
                    ffnOut[i].mul(by: gateVals.scalarAt(i))

                    }
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
        gpu.stopCapture()

       // print(thisToken, topKVector.strInt)
        
        if thisToken < 2 {
//            assert(topKVector.getInt(index: 0) == [18816, 31739][thisToken])//, 3971, 25215, 2810, 20686, 9608, 20686, 9608, 20686, 9608, 20686][thisToken])
        }
            
        if !silent {
            sumEvalTime += Date().timeIntervalSince(evalTime)
            print("prep: \(ptime, precision: 2) ms; eval: \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")
            evalTime = Date()
        }

        let topToken = Int(topKVector.getInt(index: 0))

        if tokens.count-1 == thisToken {
            let tokEmbedding = modelData.tokEmbeddings[topToken].asFloat32()
            tokens.append(tokEmbedding)
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
        print("avg eval time \(sumEvalTime*1000/Double(numTokens-2), precision: 2) ms")
        print("avg prep time \(sumPrepTime*1000/Double(numTokens-2), precision: 2) ms")
        
        print("both \((Double(numTokens-2)/(sumEvalTime+sumPrepTime)), precision: 2) tps")
        print("just eval \((Double(numTokens-2)/(sumEvalTime)), precision: 2) tps")

    }

    return archive
}

var runControl = false
silent = false
//_ = control(isTest: true, tokens: tokens, quant:0.30)
_ = runNetwork(isTest: false, tokens: tokens, quant:0.25)
_ = runNetwork(isTest: true, tokens: tokens, quant:0.25)

_ = runNetwork(isTest: false, tokens: tokens, quant:0.60)
_ = runNetwork(isTest: true, tokens: tokens, quant:0.60)

_ = runNetwork(isTest: false, tokens: tokens, quant:0.90)
_ = runNetwork(isTest: true, tokens: tokens, quant:0.90)

_ = runNetwork(isTest: false, tokens: tokens, quant:0.99)
_ = runNetwork(isTest: true, tokens: tokens, quant:0.99)






os_signpost(.end, log: log, name: "TokenGen")

exit(0)
runControl = false
_ = runControl(isTest: true, tokens: tokens, quant:0.30)

/*
for i in 2...25 {
    let _ = runNetwork(isTest: true, tokens: tokens, quant:Double(i*2)/100)
}*/
