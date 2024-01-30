//
//  main.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 24/01/2024.
//

import Foundation
import Metal

let devices = MTLCopyAllDevices()
assert(!devices.isEmpty, "No Metal devices available")

// Optionally, you can choose a device based on specific criteria.
// For simplicity, let's use the first available device.
let device = devices[0]

let res = test(device: device)
//print("done")
//exit(res)

print("loading")
let modelData = loadModelData(from: "shape.json", device: device)
let tokens = loadTokens(device: device)

print("Hello, World!")

let commandQueue = device.makeCommandQueue()!
let library = device.makeDefaultLibrary()!
let computeFunction = library.makeFunction(name: "matrixVectorMultiply")!

let dim = 4096
let dim_range = 0...4095

let headDim = 128  // Example head dimension
let numHeads = 32
let maxSeqLen = 128  // Example maximum sequence length
let freqsCis = createFreqsCis(headDim: headDim, maxSeqLen: maxSeqLen)

let tokenNum = 0

var xkLayerTokenHead = [[[Layer]]]()
var xvLayerTokenHead = [[[Layer]]]()
var xqLayerTokenHead = [[[Layer]]]()


for _ in 0...3 {
    xkLayerTokenHead.append([[Layer]]())
    xvLayerTokenHead.append([[Layer]]())
    xqLayerTokenHead.append([[Layer]]())
}

let numTokens = 1

for layerNo in 0...3 { //modelData.layers {
    var h = tokens[0]
    let layer = modelData.layers[layerNo]!
    let wa = layer["attention_norm"]!
    let wq = layer["attention.wq"]!
    let wk = layer["attention.wk"]!
    let wv = layer["attention.wv"]!
    let wo = layer["attention.wo"]!
    print(h[0])
    print(h[1])
    print(h[2])
    print(h[3])
    print(h[4])
    print("hhhh")

    let h_norm = rms_norm(layer: h)

//    print(wa.shape)
    let xn = mul(vec: h_norm, by:wa)

    let xq = mul_col(vec: xn, by: wq)
    let xk = mul_col(vec: xn, by: wk)
    let xv = mul_col(vec: xn, by: wv)
    
    var xq_heads = reshape(vec: xq, newDimSize: headDim)
    var xk_heads = reshape(vec: xk, newDimSize: headDim)
    let xv_heads = reshape(vec: xv, newDimSize: headDim)
    
    for i in 0..<numHeads {
        xq_heads[i] = multiplyLayerByComplexArray(layer: xq_heads[i], complexArray: freqsCis[tokenNum])
        xk_heads[i] = multiplyLayerByComplexArray(layer: xk_heads[i], complexArray: freqsCis[tokenNum])
    }

    xkLayerTokenHead[layerNo].append(xk_heads)
    xvLayerTokenHead[layerNo].append(xv_heads)
    xqLayerTokenHead[layerNo].append(xq_heads)
    
    let xkTokenHeads = xkLayerTokenHead[layerNo]
    let xvTokenHeads = xvLayerTokenHead[layerNo]
    let xqTokenHeads = xqLayerTokenHead[layerNo]

    var scores = [[[Float16]]]()
    for head in 0..<numHeads {
        scores.append([[Float16]]())
        for t1 in 0..<numTokens {
            scores[head].append([Float16]())
            for _ in 0..<numTokens {
                scores[head][t1].append(-100)
            }
        }
    }

    // calculate scores
    //scores = np.matmul(xk, xq, axes=[(0,2),(2,0),(2,1)]) / np.sqrt(head_dim)

    for t1 in 0..<numTokens {
        for t2 in 0..<numTokens {
            for headNo in 0..<numHeads {
                var sum: Float16 = 0.0;
                for i in 0..<headDim {
                    sum += xkTokenHeads[t2][headNo][i] * xqTokenHeads[t1][headNo][i]
                }
                scores[headNo][t1][t2] = sum / sqrt(Float16(headDim))
            }
        }
    }
    
    // masking scores 0 for lower layers
    for headNo in 0..<numHeads {
        for t1 in 0..<numTokens {
            for t2 in t1+1..<numTokens {
                scores[headNo][t1][t2] -= 10000.0
            }
        }
    }
    
    // softmax
    for headNo in 0..<numHeads {
        for t1 in 0..<numTokens {
            var maxVal = Float16(0.0)
            var sum = Float16(0.0)
            
            for t2 in 0..<numTokens {
                if scores[headNo][t1][t2]>maxVal {
                    maxVal = scores[headNo][t1][t2]
                }
            }
            
            for t2 in 0..<numTokens {
                let v = Float16(exp(Double(scores[headNo][t1][t2] - maxVal)))
                scores[headNo][t1][t2] = v
                sum += v
            }
            
            for t2 in 0..<numTokens {
                scores[headNo][t1][t2] /= sum
            }
        }
    }
    
    var out = [[[(Float16)]]]()
    
    for tok1 in 0..<numTokens {
        out.append([[(Float16)]]())
        for headNo in 0..<numHeads {
            out[tok1].append([(Float16)]())
            for i in 0..<headDim {
                var suma: Float16 = 0.0
                for tok2 in 0..<numTokens {
                    suma += scores[headNo][tok1][tok2] * xvTokenHeads[tok1][headNo][i]
                }
                out[tok1][headNo].append(suma)
            }
        }
    }
    
    // merge heads
    var output = [[Float16]]()
    for tok1 in 0..<numTokens {
        output.append([Float16]())
        for headNo in 0..<numHeads {
            for i in 0..<headDim {
                output[tok1].append(out[tok1][headNo][i])
            }
        }
    }
    
    // ffn output
    let attnOutput = createLayer(from: output[0], using: device)
    let attnFfn = mul_row(vec:attnOutput, by: wo)

    assert_vec(layer: h, mul:100, val:[0.02, -0.01, 0.01, 0.02, -0.01])
    assert_vec(layer: attnFfn, mul: 100, val:[-0.05, -0.02, -0.09, -0.07, -0.04])
    
    add(dest: &h, by: attnFfn)
    assert_vec(layer: h, mul:100, val:[-0.03, -0.03, -0.07, -0.04, -0.05])
    
    let h_norm2 = rms_norm(layer: h)
    assert_vec(layer: h_norm2, mul:100, val:[-0.75, -0.68, -1.72, -0.944, -1.26])

    let wn = layer["ffn_norm"]!
    let w1 = layer["feed_forward.w1"]!
    let w2 = layer["feed_forward.w2"]!
    let w3 = layer["feed_forward.w3"]!

    let fxn = mul(vec: h_norm2, by:wn)
    assert_vec(layer: fxn, mul:100, val:[-0.04, -0.06, -0.14, -0.07, -0.09])
    
    let fx1 = mul_col(vec: fxn, by: w1)
    let fx3 = mul_col(vec: fxn, by: w3)
    
    assert_vec(layer: fx1, mul:100, val:[-0.1, -0.05, -0.08, -0.13, 0.11])
    
    //    x = ((x1 / (1.0 + np.exp(-x1))) * x3
    var x = [(Float16)]()
    assert(fx3.shape[0] == 11008)
    for i in 0..<fx3.shape[0] {
        let val: Double = Double(fx1[i])/(1+exp(Double(-fx1[i]))) * Double(fx3[i])
        x.append(Float16(val))
    }

    let fx = createLayer(from: x, using: device)
    assert_vec(layer:fx, mul: 10000, val:[-0.0008, -0.0016, 0.0019, -0.0055, 0.0008])
    let fx2 = mul_row(weights:w2, by:fx)
    assert_vec(layer:fx2, mul:100, val:[-0.03, -0.09, 0.03, -0.05, 0.06])
    
    add(dest: &h, by: fx2)
    func assert_vec(layer: Layer, mul: Int, val: [Float16]) {
        for i in 0..<val.count {
            if round(layer[i]*Float16(mul)) != round(val[i]*Float16(mul)) {
                print("assert failed for values")
                for j in 0..<val.count {
                    print(layer[j])
                }
                fatalError("assert failed, on pos \(i), \(layer[i]) ≠ \(val[i])")
            }
        }
    }
    assert_vec(layer:h, mul:100, val:[-0.06,-0.12,-0.05,-0.09,0.01,-0.01,-0.07])
    exit(0)

}

print("done")
exit(0)

let pipelineState = try device.makeComputePipelineState(function: computeFunction)

let vector = Array(repeating: Float(0.0), count: 4096)
let matrix = Array(repeating: Array(repeating: Float(0.0), count: 10096), count: 4096)


let matrixBuffer = modelData.layers[0]!["feed_forward.w1"]!.buffer

//device.makeBuffer(bytes: matrix, length: matrix.count * MemoryLayout<Float>.size, options: .storageModeShared)
let vectorBuffer = device.makeBuffer(bytes: vector, length: vector.count * MemoryLayout<Float>.size, options: .storageModeShared)
let resultBuffer = device.makeBuffer(length: 10096 * MemoryLayout<Float>.size, options: .storageModeShared)

// dispatch

let gridSize = MTLSize(width: 10096, height: 1, depth: 1)
let threadGroupSize = MTLSize(width: min(pipelineState.threadExecutionWidth, 10096), height: 1, depth: 1)

let commandBuffer = commandQueue.makeCommandBuffer()!
let commandEncoder = commandBuffer.makeComputeCommandEncoder()!

commandEncoder.setComputePipelineState(pipelineState)
commandEncoder.setBuffer(matrixBuffer, offset: 0, index: 0)
commandEncoder.setBuffer(vectorBuffer, offset: 0, index: 1)
commandEncoder.setBuffer(resultBuffer, offset: 0, index: 2)

// Dispatch the compute command
commandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)

commandEncoder.endEncoding()

let startTime = Date()

commandBuffer.commit()
commandBuffer.waitUntilCompleted()

let endTime = Date()
let timeInterval = endTime.timeIntervalSince(startTime)

print("Average execution time for 1 run: \(timeInterval) seconds")
    
let buffer = resultBuffer!

let data = NSData(bytesNoCopy: buffer.contents(), length: 10096 * MemoryLayout<Float>.size, freeWhenDone: false)
var resultArray = [Float](repeating: 0, count: 10096)
data.getBytes(&resultArray, length: resultArray.count * MemoryLayout<Float>.size)
