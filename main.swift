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
print("starting up")
let testLoader = TensorLoader(path: "./", model: "tests")

let log = OSLog(subsystem: "com.kolinko", category: "Performance")
 
let gpu = Gpu()
print("loading")

//runConvert([.mixtral, .q8])
//exit(0)

let stateDim = 4096
let hiddenDim = 14336
let goQ8 = false
let percentLoad = goQ8 ? 0x8 : 0xC // from 0 to max binSize
let bSize: Int

var numLayers = 10
var numExperts = 2
var numTokens = 100
let goVerify = (numLayers == 10 && numExperts == 2)

let modelData = Model(numLayers: numLayers, numExperts: numExperts, percentLoad: percentLoad)

let t = Tokeniser(modelData)

//var tokens = [VectorFloat]()
let tokens = t.embed([1, 1602, 460])

os_signpost(.end, log: log, name: "Loading")

let headDim = 128  // Example head dimension
let numHeadsKV = 8
let numHeads = 32
let kvRepeats : Int = numHeads/numHeadsKV
let maxSeqLen = 2048
let maxTokens = maxSeqLen
let freqsCis = createFreqsCis(headDim: headDim, maxSeqLen: maxSeqLen)

//modelRunTests()

//modelProfile()

print()
gpu.eval()

var silent = true

func runNetwork(isTest: Bool, tokens _tokens: [VectorFloat], quant: Double = 1.0) {
    var tokens = _tokens
    let xkLayerTokenHead = Matrix4DFloat(shape:[numLayers, maxTokens, numHeads, headDim])
    let xvLayerToken = Matrix3DFloat(shape:[numLayers, maxTokens, stateDim])
    
    var h : VectorFloat = tokens[0]
    
    let x1 = VectorFloat(shape:[hiddenDim])
    let x3 = VectorFloat(shape:[hiddenDim])
    let x2 = VectorFloat(shape:[hiddenDim])
    let ffnOut = [VectorFloat]([VectorFloat(shape:[stateDim]), VectorFloat(shape:[stateDim])])
    
    // buffers
    let h_norm = VectorFloat(shape:[stateDim])
    let attnOutMatrix = MatrixFloat(shape: [numHeads, headDim])
    let fxn = VectorFloat(shape:[stateDim])
    let gateOut = VectorFloat(shape: [numExperts])
    let gateIdxs = VectorFloat(shape:[2])
    let gateVals = VectorFloat(shape:[2])

    
    // timers
    var output = ""
    gpu.warnOfEvals = false
    let finalEvalTime = Date()
    var evalTime = Date()
    var sumPrepTime = Date().timeIntervalSince(evalTime)
    var sumEvalTime = Date().timeIntervalSince(evalTime)

    
    for thisToken in 0...numTokens {
        let scores = MatrixFloat(shape: [numHeads, thisToken+1])

        if thisToken == 2 {
            gpu.eval()
            sumPrepTime = Date().timeIntervalSince(evalTime)
            sumEvalTime = Date().timeIntervalSince(evalTime)
        }
        
        h = tokens[thisToken].copy()

        for layerNo in 0..<numLayers {
            let layer = modelData.layers[layerNo]!
            h.rmsNorm(out: h_norm)
            
            h_norm.mul(by:layer.attnNorm)
            let xq = basicMul(v: h_norm, by: layer.wq.core)
            let xk = xkLayerTokenHead[layerNo][thisToken].asVector()
            let xk_temp = basicMul(v: h_norm, by: layer.wk.core)
            xk_temp.repeated(kvRepeats, into:xk)
            
            let xqHeads = xq.asMatrix(newCols: headDim)
            let xkHeads = xk.asMatrix(newCols: headDim)
            
            for i in 0..<numHeads {
                xqHeads[i].mul(complexArray: freqsCis[thisToken])
                xkHeads[i].mul(complexArray: freqsCis[thisToken])
            }
            
            let xv_temp = basicMul(v: h_norm, by: layer.wv.core)
            xv_temp.repeated(kvRepeats, into: xvLayerToken[layerNo][thisToken])
            
            let xkTokenHeads = xkLayerTokenHead[layerNo]
            let xvToken = xvLayerToken[layerNo]
            
            calcScores2(xq_heads: xqHeads, xkTokenHeads: xkTokenHeads, numTokens: thisToken+1, out: scores)
            
            for headNo in 0..<numHeads {
                scores[headNo].softmax()
            }

            sumScores(numHeads: numHeads, headDim:headDim, scores: scores, xvToken: xvToken, numTokens: thisToken+1, out: attnOutMatrix)
            let attnOutput = attnOutMatrix.asVector()

            let attnFfnOut = basicMul(v: attnOutput, by: layer.wo.core)
            
            h.add(by: attnFfnOut)

            h.rmsNorm(out:fxn)
            
            fxn.mul(by:layer.ffnNorm)
            basicMul(v:fxn, by:layer.ffnGate, out:gateOut)
            mpsTopK(v: gateOut, topK: 2, outIndexes: gateIdxs, outValues: gateVals)
            
            gateVals.softmax()
            for i in 0..<2 {
                let expIdx = gateIdxs.scalarAt(i)
                if !goQ8 {
                    expertMul(v: fxn, by: layer.w1, expNo: expIdx, out: x1, quant: quant)
                    expertMul(v: fxn, by: layer.w3, expNo: expIdx, out: x3, quant: quant)
                    
                    silu(x1, x3, out: x2)
                    expertMul(v: x2, by: layer.w2, expNo: expIdx, out: ffnOut[i], quant: quant)
                    ffnOut[i].mul(by: gateVals.scalarAt(i))
                } else {
                    expertMulQ8(v: fxn, by: layer.w1, expNo: expIdx, out: x1, quant: quant)
                    expertMulQ8(v: fxn, by: layer.w3, expNo: expIdx, out: x3, quant: quant)
                    
                    silu(x1, x3, out: x2)
                    expertMulQ8(v: x2, by: layer.w2, expNo: expIdx, out: ffnOut[i], quant: quant)
                    ffnOut[i].mul(by: gateVals.scalarAt(i))
                }
            }
            
            h.add(by: ffnOut[0])
            h.add(by: ffnOut[1])
            
            gpu.eval()
            
            testVec("h-out:\(thisToken):\(layerNo)", h)
        }
        
        testVec32("token:\(thisToken)", h)
        
        let outNormed = VectorFloat(shape: [stateDim])
//        let outNormed = h.rmsNormed()
        h.rmsNorm(out: outNormed)
        outNormed.mul(by: modelData.norm.asVector())
        
        let outputVector = VectorFloat(shape:[modelData.output.outSize])
        basicMul(v: outNormed, by: modelData.output.core, out: outputVector)
        
        testVec32("ovector:\(thisToken)", outputVector)
        testReport(thisToken >= 10)
        
        let topKVector = mpsTopK(v: outputVector)
        
        sumPrepTime += Date().timeIntervalSince(evalTime)
        let ptime = Date().timeIntervalSince(evalTime)*1000
        evalTime = Date()
        
        if tokens.count-1 == thisToken {
            tokens.append(modelData.tokEmbeddings.fetchRow(topKVector.scalarAt(0)))
            gpu.eval()
            let topToken = Int(topKVector.getInt(index: 0))
            let newS = t[topToken].replacingOccurrences(of: "▁", with: " ")
            output += newS
            if (silent) {
                print(newS, terminator: goVerify ? "\n" : "")
                fflush(stdout)
                if output.contains("</s>") {
                    break
                }
            }
        }
        sumEvalTime += Date().timeIntervalSince(evalTime)
        evalTime = Date()
        
        if !silent {
            print("prep: \(ptime, precision: 2) ms; eval: \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")
            evalTime = Date()
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
        print("avg eval time \(sumEvalTime*1000/Double(tokens.count-2), precision: 2) ms")
        print("avg prep time \(sumPrepTime*1000/Double(tokens.count-2), precision: 2) ms")
        
        print("both \((Double(tokens.count-2)/(sumEvalTime+sumPrepTime)), precision: 2) tps")
        print("just eval \((Double(tokens.count-2)/(sumEvalTime)), precision: 2) tps")
        
    } else {
        print("\n")
        let evalTime = sumEvalTime*1000/Double(tokens.count-2)
        let prepTime = sumPrepTime*1000/Double(tokens.count-2)
        let sumTps = Double(tokens.count-2)/(sumEvalTime+sumPrepTime)
        let evalTps = Double(tokens.count-2)/(sumEvalTime)
        print("\(quant*100, precision: 2)%: \(prepTime, precision: 2)ms + \(evalTime, precision: 2)ms (\(evalTps, precision: 2)/\(sumTps, precision: 2)tps)")
        print("")
    }
    
}

var runControl = false
silent = true
runNetwork(isTest: true, tokens: tokens, quant:1)

var storedIntegers: [Int] = []
var storedStrings: [String] = []

var quant: Double = 0.25

while true {
    print("Enter 'p XX' to store a number or any text to store it as a string ('q' to quit):")
    while true {
        print("> ", terminator: "")
        if let input = readLine() {
            if let number = Int(input), (0...100).contains(number) {
                quant = Double(number)/100.0
            } else {
                let tokens = t.embed("[INST]"+input+"[/INST]")
                runNetwork(isTest: true, tokens: tokens, quant:quant)
            }
        }
    }
}
