//
//  runNetwork.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 13/04/2024.
//

import Foundation


struct Reply {
    let reply: String
    let hitMiss: [Int]
}

func runNetwork(isTest: Bool = false, tokens _tokens: [VectorFloat], effort _effort: Double = 1.0, srcTokenIds : [Int]? = nil, limitLogits : [Int]? = nil) -> Reply {
    let effort = _effort
    let bm1 = BucketMulFaster()
    let bm2 = BucketMulFaster()
    let bm3 = BucketMulFaster()

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
    let attnOutput = VectorFloat(shape: [numHeads * headDim])
    let fxn = VectorFloat(shape:[stateDim])
    let gateOut = VectorFloat(shape: [numExperts])
    let gateIdxs = VectorFloat(shape:[2])
    let gateVals = VectorFloat(shape:[2])
    let outputVector = VectorFloat(shape:[modelData.output.outSize])
    let outNormed = VectorFloat(shape: [stateDim])
    
    // timers
    var output = ""
    gpu.warnOfEvals = false
    let finalEvalTime = Date()
    var evalTime = Date()
    var sumPrepTime = Date().timeIntervalSince(evalTime)
    var sumEvalTime = Date().timeIntervalSince(evalTime)

    let _layer = modelData.layers[0]!
    let xq = VectorFloat(shape: [_layer.wq.outSize])
    let xq_temp = VectorFloat(shape: [_layer.wq.outSize])
    let xk_temp = VectorFloat(shape: [_layer.wk.outSize])
    let xk_temp2 = VectorFloat(shape: [_layer.wk.outSize*4])
    let xv_temp = VectorFloat(shape: [_layer.wk.outSize])
    let attnFfnOut = VectorFloat(shape: [_layer.wo.outSize])
    
    var hitMiss = [Int]()
    
    for thisToken in 0...numTokens {
        let scores = MatrixFloat(shape: [numHeads, thisToken+1])

        if thisToken == 2 {
            gpu.eval(noWarn: true)
            // first token is a warmup, so we're not benchmarking it
            sumPrepTime = Date().timeIntervalSince(evalTime)
            sumEvalTime = Date().timeIntervalSince(evalTime)
            gpu.warnOfEvals = true
        }
        
        h = tokens[thisToken].copy()
        gpu.warnOfEvals = false
        gpu.eval()
        gpu.warnOfEvals = true
//        gpu.startCapture()
        for layerNo in 0..<numLayers {
            //gpu.startCapture()

            let layer = modelData.layers[layerNo]!
            testVec("h_in:\(thisToken):\(layerNo)", h)

            h.rmsNormFast(out: h_norm)
            testVec("h_norm_rms:\(thisToken):\(layerNo)", h_norm)
            h_norm.mul(by:layer.attnNorm)
            testVec("h_norm_mul:\(thisToken):\(layerNo)", h_norm)

            let xk = xkLayerTokenHead[layerNo][thisToken].asVector()
            let xv = xvLayerToken[layerNo][thisToken]
            let expIdx = ScalarFloat(value: 0)

            gpu.concurrent([{
                bm1.findCutoff(v: h_norm, eWeights: layer.wq, expNo: expIdx, effort: effort)
                bm2.findCutoff(v: h_norm, eWeights: layer.wk, expNo: expIdx, effort: effort)
                bm3.findCutoff(v: h_norm, eWeights: layer.wv, expNo: expIdx, effort: effort)
            }, {
                bm1.prepareDispatch(v: h_norm, eWeights: layer.wq, expNo: expIdx, effort: effort)
                bm2.prepareDispatch(v: h_norm, eWeights: layer.wk, expNo: expIdx, effort: effort)
                bm3.prepareDispatch(v: h_norm, eWeights: layer.wv, expNo: expIdx, effort: effort)
            }, {
                bm1.mul(by: layer.wq, out: xq)
                bm2.mul(by: layer.wk, out: xk_temp)
                bm3.mul(by: layer.wv, out: xv_temp)
             },{
                bm1.reintegrate(out: xq_temp)
                bm2.reintegrate(out: xk_temp)
                bm3.reintegrate(out: xv_temp)
             }])

            xk_temp.repeated(kvRepeats, into:xk_temp2)
            xv_temp.repeated(kvRepeats, into:xv)
            
            let xqHeads = xq.asMatrix(newCols: headDim)
            let xkHeads = xk.asMatrix(newCols: headDim)

            let xkTokenHeads = xkLayerTokenHead[layerNo]
            let xvToken = xvLayerToken[layerNo]
            
            let fCis = freqsCis[thisToken]
            testVec("xqHeads:\(thisToken):\(layerNo)", xqHeads.asVector())
            
            gpu.concurrent([{
                gpu.deploy("rope_mx", buffers: [xq_temp, fCis, xqHeads], ints:[xqHeads.cols], threadCount: [xqHeads.cols, xqHeads.rows])
                gpu.deploy("rope_mx", buffers: [xk_temp2, fCis, xkHeads], ints:[xkHeads.cols], threadCount: [xkHeads.cols, xkHeads.rows])
            }])
                           
//            xqHeads.rope(complexArray: fCis)
//            xkHeads.rope(complexArray: fCis)
            testVec("xqHeadsROPE:\(thisToken):\(layerNo)", xqHeads.asVector())
            

            calcScores2(xq_heads: xqHeads,
                        xkTokenHeads: xkTokenHeads,
                        numTokens: thisToken+1,
                        out: scores)

            testVec("scores:\(thisToken):\(layerNo)", scores.asVector())

            scores.softmax()
            testVec("scoresSM:\(thisToken):\(layerNo)", scores.asVector())

            sumScores(scores: scores,
                      xvToken: xvToken,
                      numTokens: thisToken+1,
                      out: attnOutput)
            
            expertMul(v: attnOutput, by: layer.wo, out: attnFfnOut, effort: effort)
            
            testVec("attnFfnOut:\(thisToken):\(layerNo)", attnFfnOut)

            h.add(by: attnFfnOut)
            h.rmsNormFast(out:fxn)
            
            fxn.mul(by:layer.ffnNorm)

            testVec("fxn:\(thisToken):\(layerNo)", fxn)
            
            if layer.ffnGate == nil {
                let expIdx = ScalarFloat(value: 0)

                gpu.concurrent([{
                    bm1.findCutoff(v: fxn, eWeights: layer.w1, expNo: expIdx, effort: effort)
                    bm2.findCutoff(v: fxn, eWeights: layer.w3, expNo: expIdx, effort: effort)
                }, {
                    bm1.prepareDispatch(v: fxn, eWeights: layer.w1, expNo: expIdx, effort: effort)
                    bm2.prepareDispatch(v: fxn, eWeights: layer.w3, expNo: expIdx, effort: effort)
                }, {
                    bm1.mul(by: layer.w1, out: x1)
                    bm2.mul(by: layer.w3, out: x3)
                },{
                    bm1.reintegrate(out: x1)
                    bm2.reintegrate(out: x3)
                }])
                
                silu(x1, x3, out: x2)
                expertMul(v: x2, by: layer.w2, expNo: expIdx, out: ffnOut[0], effort: effort)
                h.add(by: ffnOut[0])
            } else {
                basicMul(v:fxn, by:layer.ffnGate!, out:gateOut)
                mpsTopK(v: gateOut, topK: 2, outIndexes: gateIdxs, outValues: gateVals)
                
                gateVals.softmax()
                for i in 0..<2 {
                    let expIdx = gateIdxs.scalarAt(i)
                    expertMul(v: fxn, by: layer.w1, expNo: expIdx, out: x1, effort: effort)
                    expertMul(v: fxn, by: layer.w3, expNo: expIdx, out: x3, effort: effort)
                    
                    silu(x1, x3, out: x2)
                    expertMul(v: x2, by: layer.w2, expNo: expIdx, out: ffnOut[i], effort: effort)
                    ffnOut[i].mul(by: gateVals.scalarAt(i))
                }
                
                h.add(by: ffnOut[0])
                h.add(by: ffnOut[1])
            }
            
            
            testVec("h-out:\(thisToken):\(layerNo)", h)
            gpu.stopCapture()

        }

        h.rmsNormFast(out: outNormed)
        outNormed.mul(by: modelData.norm)
        
        basicMul(v: outNormed, by: modelData.output.core, out: outputVector)

        gpu.warnOfEvals = false
        
        sumPrepTime += Date().timeIntervalSince(evalTime)
        let ptime = Date().timeIntervalSince(evalTime)*1000
        evalTime = Date()
        
        gpu.eval()
        
        sumEvalTime += Date().timeIntervalSince(evalTime)
        evalTime = Date()

        testVec32("token:\(thisToken)", h)
        testVec32("ovector:\(thisToken)", outputVector)

        let topKVector = mpsTopK(v: outputVector)

        if tokens.count-1 == thisToken && limitLogits != nil {
            gpu.eval()
            for i in 0..<topKVector.count {
                for j in 0..<limitLogits!.count {
                    if limitLogits![j] == topKVector.getLong(index: i) {
                        //print("answer:", j+1)
                        return Reply(
                            reply: String(j+1),
                            hitMiss: []
                        )
                    }
                }
            }
        }
        
        if tokens.count-1 == thisToken {
            tokens.append(modelData.tokEmbeddings.fetchRow(topKVector.scalarAt(0)))
            gpu.eval()

            if thisToken >= 30 && goVerify {
                speedReport()
            }

            testReport(thisToken >= 10)
            if thisToken >= 10 && goVerify {
                exit(0)
            }

            let topToken = Int(topKVector.getInt(index: 0))
            let topT = t[topToken].replacingOccurrences(of: "▁", with: " ")
            let newS = topT.replacingOccurrences(of: "<0x0A>", with: "↩")
            output += topT
            if (silent) {
                print(newS, terminator: goVerify ? "\n" : "")
                fflush(stdout)
                if output.contains("</s>") {
                    break
                }
            }
        }
        
        if srcTokenIds == nil || thisToken >= srcTokenIds!.count-1 {
            gpu.eval()
            hitMiss.append(Int(topKVector.scalarAt(0).intVal))
        }
        gpu.eval()
        gpu.warnOfEvals = false
        
        if !silent {
            print("prep: \(ptime, precision: 2) ms; eval: \(Date().timeIntervalSince(evalTime)*1000, precision: 2) ms")
            evalTime = Date()
        }
        gpu.warnOfEvals = true

        if thisToken == numTokens {
            print(" »")
            break
        }
    }
    
    evalTime = Date()
    
    speedReport()
    
    func speedReport() {
        print("\n")
        let numTkns = Double(min(tokens.count, numTokens)-2)
        var _sumEvalTime = sumEvalTime
        var _sumPrepTime = sumPrepTime
        var _evalTime = _sumEvalTime*1000/numTkns
        var _prepTime = _sumPrepTime*1000/numTkns
        
        _evalTime /= Double(numLayers) / 32.0
        _prepTime /= Double(numLayers) / 32.0
        
        _sumEvalTime /= Double(numLayers) / 32.0
        _sumPrepTime /= Double(numLayers) / 32.0
        
        
        let sumTps = numTkns/(_sumEvalTime+_sumPrepTime)
        let evalTps = numTkns/(_sumEvalTime)
        
        print("\(effort*100, precision: 2)%: prep: \(_prepTime, precision: 2)ms ; eval: \(_evalTime, precision: 2)ms (\(evalTps, precision: 2) » \(sumTps, precision: 2)tps)")
        print("")
                
    }
    
    return Reply(
        reply: output,
        hitMiss: (srcTokenIds != nil ? srcTokenIds! : []) + hitMiss
    )
}
