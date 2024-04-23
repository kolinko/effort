//
//  runNetwork.swift
//  effort
//


import Foundation

/*
 reply - reply string
 hitMiss - predicted tokens, used in benchmarks
 
 the Reply structure, and the whole reply of runNetwork needs a serious cleanup and refactor.
 
 */

struct Reply {
    let reply: String
    let hitMiss: [Int]
}

/*
 
 this obviously needs refactoring :D
 
*/

let xkLayerTokenHead = Matrix4DFloat(shape:[numLayers, maxTokens, numHeads, headDim])
let xvLayerToken = Matrix3DFloat(shape:[numLayers, maxTokens, stateDim])
let x1 = VectorFloat(shape:[hiddenDim])
let x3 = VectorFloat(shape:[hiddenDim])
let x2 = VectorFloat(shape:[hiddenDim])
let ffnOut = [VectorFloat]([VectorFloat(shape:[stateDim]), VectorFloat(shape:[stateDim])])
let h_norm = VectorFloat(shape:[stateDim])
let attnOutput = VectorFloat(shape: [numHeads * headDim])
let fxn = VectorFloat(shape:[stateDim])
let gateOut = VectorFloat(shape: [numExperts])
let gateIdxs = VectorFloat(shape:[2])
let gateVals = VectorFloat(shape:[2])
let outputVector = VectorFloat(shape:[modelData.output.rows])
let outNormed = VectorFloat(shape: [stateDim])
let _layer = modelData.layers[0]!
let xq = VectorFloat(shape: [_layer.wq.outSize])
let xq_temp = VectorFloat(shape: [_layer.wq.outSize])
let xk_temp = VectorFloat(shape: [_layer.wk.outSize])
let xk_temp2 = VectorFloat(shape: [_layer.wk.outSize*4])
let xv_temp = VectorFloat(shape: [_layer.wk.outSize])
let attnFfnOut = VectorFloat(shape: [_layer.wo.outSize])
let expIdxZero = ScalarFloat(value: 0)
let scores = MatrixFloat(shape: [numHeads, maxTokens])

/*
 
 The main loop.
 
 If you worked with any other LLM model implementation,
 it should be quite understandable, but ugly.
 
 I prefer to have this done this way, not the regular way it's implemented
 within the classes, because it's easier to see every single multiplication
 implemented one by one, and debug / profile / improve them.
 
 Regular implementations - like the HF or MLX one, make it very difficult to profile
 and hide the calculations deep within the class/object layer.
 
 */

func runNetwork(tokens _tokens: [VectorFloat],
                effort: Double = 1.0,
                srcTokenIds : [Int]? = nil,
                limitLogits : [Int]? = nil,
                returnPredictions: Bool = false) -> Reply {

    let tokens = MatrixFloat(shape: [maxTokens, stateDim]).asVectorList()
    for i in 0..<_tokens.count {
        tokens[i].copyFrom(_tokens[i])
    }
    gpu.eval()
    
    var h : VectorFloat = tokens[0]
   
    var output = ""
    var evalTime = Date()
    var sumPrepTime = Date().timeIntervalSince(evalTime)
    var sumEvalTime = Date().timeIntervalSince(evalTime)
    
    var countTokens = 0
    var predictedTokens = [Int]()
    
    /*
     
     I beg any Metal engineer to help me here.
     The core loop has delays between kernel invocations that I couldn't get rid of.
     
     Even with effort = 0, it still takes 15ms on my machine to generate a single token.
     I know it can be fixed, because Llama.cpp has no such delays.
     
     - On GPU profiler, it shows blank spaces between some kernel invocations.
     - I tried using concurrent encoders, but that gave no improvement.
     - Perhaps it's because buffers are hazard-tracked, and we should directly implement fences?
     
     I beg, again, someone to help out here. This is the main unneccessary slowdown of the whole engine.
     
     Also:
     - it seems something is subtly broken in one of the funcs - perhaps rmsnorm, or scores?
       because the Mistral model doesn't work as good as it used to be, and Mixtral works barely at all.
       I think bucketMul is doing it's job correctly, but one of the other funcs has other params
     
     */
    
    var firstTok = true
    
    for thisToken in 0...numTokens {
        countTokens += 1
        
        if thisToken == 2 {
            gpu.eval(noWarn: true)
            sumPrepTime = Date().timeIntervalSince(evalTime)
            sumEvalTime = Date().timeIntervalSince(evalTime)
        }
        
        h = tokens[thisToken]

        for layerNo in 0..<numLayers {
            let layer = modelData.layers[layerNo]!
            h.rmsNormFast(out: h_norm)
            h_norm.mul(by:layer.attnNorm)

            let xk = xkLayerTokenHead[layerNo][thisToken].asVector()
            let xv = xvLayerToken[layerNo][thisToken]
            
            expertMul(v: h_norm, by: layer.wq, out: xq_temp, effort: effort)
            expertMul(v: h_norm, by: layer.wk, out: xk_temp, effort: effort)
            expertMul(v: h_norm, by: layer.wv, out: xv_temp, effort: effort)

            xk_temp.repeated(kvRepeats, into:xk_temp2)
            xv_temp.repeated(kvRepeats, into:xv)
            
            let xqHeads = xq.asMatrix(newCols: headDim)
            let xkHeads = xk.asMatrix(newCols: headDim)

            let xkTokenHeads = xkLayerTokenHead[layerNo]
            let xvToken = xvLayerToken[layerNo]
            
            let fCis = freqsCis[thisToken]
            
            gpu.concurrent([{
                gpu.deploy("rope_mx", buffers: [xq_temp, fCis, xqHeads], ints:[xqHeads.cols], threadCount: [xqHeads.cols, xqHeads.rows])
                gpu.deploy("rope_mx", buffers: [xk_temp2, fCis, xkHeads], ints:[xkHeads.cols], threadCount: [xkHeads.cols, xkHeads.rows])
            }])
            // ^ concurrent gives nothing here, should remove it, but don't want to change it right before premiere
            
            scores.shape = [numHeads, thisToken+1]
            calcScores(xq_heads: xqHeads,
                        xkTokenHeads: xkTokenHeads,
                        numTokens: thisToken+1,
                        out: scores)

            scores.softmax()

            sumScores(scores: scores,
                      xvToken: xvToken,
                      numTokens: thisToken+1,
                      out: attnOutput)
            
            // ^ these funcs are not optimised at all, and I think they cause the main slowdown with longer context
            // first step would be to just fix it using the regular optimisation techniques
            // second step would be inventing sth like BucketMul, but for sumscores/calcscores
            
            expertMul(v: attnOutput, by: layer.wo, out: attnFfnOut, effort: effort)
            
            h.add(by: attnFfnOut)
            h.rmsNormFast(out:fxn)
            
            fxn.mul(by:layer.ffnNorm)

            if layer.ffnGate == nil {
                expertMul(v: fxn, by: layer.w1, expNo: expIdxZero, out: x1, effort: effort)
                expertMul(v: fxn, by: layer.w3, expNo: expIdxZero, out: x3, effort: effort)

                silu(x1, x3, out: x2)
                expertMul(v: x2, by: layer.w2, expNo: expIdxZero, out: ffnOut[0], effort: effort)
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
                        
            gpu.stopCapture()
        }

        h.rmsNormFast(out: outNormed)
        outNormed.mul(by: modelData.norm)
        
        basicMul(v: outNormed, by: modelData.output, out: outputVector)
        // ^ could be switched to BucketMul, but doesn't cause a big delay

        gpu.warnOfEvals = false
        
        sumPrepTime += Date().timeIntervalSince(evalTime)
        evalTime = Date()
        
        gpu.eval()
        
        sumEvalTime += Date().timeIntervalSince(evalTime)
        evalTime = Date()

        /*
         
         I apologise deeply for the code below -- TK.
         
         */
        
        if returnPredictions {
           let topKVector = mpsTopK(v: outputVector)
           gpu.eval()
           predictedTokens.append(topKVector.getLong(index: 0))
        }
        
        if thisToken >= _tokens.count-1 {
            let topKVector = mpsTopK(v: outputVector)

            if limitLogits != nil {
                gpu.eval()
                for i in 0..<topKVector.count {
                    for j in 0..<limitLogits!.count {
                        if limitLogits![j] == topKVector.getLong(index: i) {
                            return Reply(
                                reply: String(j+1),
                                hitMiss: []
                            )
                        }
                    }
                }
                return Reply(reply: String(99), hitMiss: [])
            }
            if firstTok {
                gpu.eval()
                print("out: \(topKVector.scalarAt(0).long)")
                firstTok = false
            }
            modelData.tokEmbeddings.fetchRow(topKVector.scalarAt(0), out: tokens[thisToken+1])
            gpu.eval()
        
            let topToken = Int(topKVector.getInt(index: 0))
            let topT = t[topToken].replacingOccurrences(of: "▁", with: " ")
            let newS = topT.replacingOccurrences(of: "<0x0A>", with: "↩")
            output += topT

            print(newS, terminator: goVerify ? "\n" : "")
            fflush(stdout)
        }
        
        if srcTokenIds == nil || thisToken >= srcTokenIds!.count-1 {
       //     gpu.eval()
          //  hitMiss.append(Int(topKVector.scalarAt(0).intVal))
        }
        gpu.eval()
        
        if thisToken >= 30 && goVerify {
            testReport(thisToken >= 10)
            break
        }

        if output.contains("</s>") {
            break
        }

        if thisToken == numTokens {
            print(" »")
            break
        }
        gpu.stopCapture()
    }
    
    
    do {
        print("\n")
        let numTkns = Double(countTokens-2)
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
        hitMiss: predictedTokens
    )
}
