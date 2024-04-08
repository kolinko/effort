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
let gpu2 = Gpu()
print("loading")
//runConvert([.mixtral, .q8])
//exit(0)

var numLayers = 32
var numExperts = 8//8

var numTokens = 100

let bam = BufferActivityManager()
bam.startPeriodicDispatch()
let modelData = Model(from: "shape.json", numLayers: numLayers, numExperts: numExperts, percentLoad: 0x8)//0xC)//C)//0x0C)//0x0C)

var tokens = [VectorFloat]()
let tokIds = [1, 1602, 460] // "How are"

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
            sumPrepTime = Date().timeIntervalSince(evalTime)
            sumEvalTime = Date().timeIntervalSince(evalTime)
        }
        
        if thisToken == 2 {
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
            
            basicMul(v: h_norm, by: layer.wv.core).repeated(kvRepeats, into: xvLayerToken[layerNo][thisToken])

            let xkTokenHeads = xkLayerTokenHead[layerNo]
            let xvToken = xvLayerToken[layerNo]
            
            let scores = calcScores2(xq_heads: xqHeads, xkTokenHeads: xkTokenHeads, numTokens: thisToken+1)
            
            if numExperts == 2 && numLayers == 2 {
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
            
            h.add(by: attnFfnOut)

            let fxn = h.rmsNormed()

            fxn.mul(by:layer.ffnNorm)
            let gateOut = VectorFloat(shape: [numExperts])
            basicMul(v:fxn, by:layer.ffnGate, out:gateOut)
            let gateIdxs = VectorFloat(shape:[2])
            let gateVals = VectorFloat(shape:[2])
            mpsTopK(v: gateOut, topK: 2, outIndexes: gateIdxs, outValues: gateVals)

            gateVals.softmax()
            for i in 0..<2 {
                let expIdx = gateIdxs.scalarAt(i)
                if !isTest {
                    expertMul3(v: fxn, by: layer.w1, expNo: expIdx, out: x1, quant: quant)
                    expertMul3(v: fxn, by: layer.w3, expNo: expIdx, out: x3, quant: quant)
                    
                    silu(x1, x3, out: x2)
                    expertMul3(v: x2, by: layer.w2, expNo: expIdx, out: ffnOut[i], quant: quant)
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
            
            let ht = h.copy().asFloat16()
            gpu.eval()

            testVec("h-out:\(thisToken):\(layerNo)", h)
            /*
            if (numLayers == 10 && numExperts == 2) {
                let tt = (testLoader["h-out:\(thisToken):\(layerNo)"] as! Vector).asFloat32()
                print(tt.cosineSimilarityTo(h))
                assert(tt.cosineSimilarityTo(h) > 0.99)
            }*/
            
        }
        
        testVec32("token:\(thisToken)", h)

        /*
        if (numLayers == 10 && numExperts == 2) {
            let tt = (testLoader[] as! VectorFloat)
            print(tt.cosineSimilarityTo(h))
            assert(tt.cosineSimilarityTo(h) > 0.99)
        }*/

        let outNormed = h.rmsNormed()
        outNormed.mul(by: modelData.norm.asVector())

        let outputVector = VectorFloat(shape:[modelData.output.outSize])
        basicMul(v: outNormed, by: modelData.output.core, out: outputVector)

        
        testVec32("ovector:\(thisToken)", outputVector)

        if (numLayers == 10 && numExperts == 2) {
            if thisToken >= 10 {
                exit(0)
            }
        }

        
        /*
        let ho = outputVector.copy()
        gpu.eval()
        testSaver[0]["ovector:\(thisToken)"] = ho
         */

        let topKVector = mpsTopK(v: outputVector)

        sumPrepTime += Date().timeIntervalSince(evalTime)
        let ptime = Date().timeIntervalSince(evalTime)*1000
        evalTime = Date()

        /*
        if thisToken < 2 {
            assert(topKVector.getInt(index: 0) == [18816, 31739][thisToken])//, 3971, 25215, 2810, 20686, 9608, 20686, 9608, 20686, 9608, 20686][thisToken])
        }*/
        


        if tokens.count-1 == thisToken {
            tokens.append(modelData.tokEmbeddings.fetchRow(topKVector.scalarAt(0)))
            gpu.eval()
            let topToken = Int(topKVector.getInt(index: 0))
            let newS = t[topToken].replacingOccurrences(of: "▁", with: " ")
            output += newS
            if (silent) {
                print(newS, terminator: "")
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

        //testSaver.save()
        //exit(0)

        
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

    
  //  exit(0)
    return archive
}

var runControl = false
silent = true
_ = runNetwork(isTest: true, tokens: tokens, quant:1)

//_ = control(isTest: true, tokens: tokens, quant:0.30)
//_ = runNetwork(isTest: true, tokens: tokens, quant:1)//0.25)

var storedIntegers: [Int] = []
var storedStrings: [String] = []

var quant: Double = 0.25

while true {
    print("Enter 'p XX' to store a number or any text to store it as a string ('q' to quit):")
    while true {
        print("> ", terminator: "")
        if let input = readLine() {
//            if input.lowercased() == "q" {  // Quit command
//                break
            if let number = Int(input), (0...100).contains(number) {
                quant = Double(number)/100.0
//                print("Stored \(number) as an integer.")
            } else {
                var tokens = [VectorFloat]()

                let tokEmbeddings = modelData.tokEmbeddings.asVectorList()
                let encoded = encode(prompt: "[INST]"+input+"[/INST]")
                for t in encoded {
                    tokens.append(tokEmbeddings[t].asFloat32())
                }
                _ = runNetwork(isTest: true, tokens: tokens, quant:quant)

//                _ = runNetwork(isTest: false, tokens: tokens, quant:quant)
//                storedStrings.append(input)
//                print("Stored \"\(input)\" as a string.")
            }
        }
    }
}
