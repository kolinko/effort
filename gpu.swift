//
//  gpu.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 11/03/2024.
//

import Foundation
import Metal


class Gpu {
    var commandBuffer : MTLCommandBuffer
    var encoder : MTLComputeCommandEncoder
    var captureON: Bool
    let captureManager: MTLCaptureManager
    
    let library : MTLLibrary
    var globalStates : [String: MTLComputePipelineState]
    let commandQueue : MTLCommandQueue
    let device : MTLDevice
    
    var warnOfEvals = false
    
    init() {
        let devices = MTLCopyAllDevices()
        assert(!devices.isEmpty, "No Metal devices available")
        self.device = devices[0]
        self.commandQueue = device.makeCommandQueue()!
        self.commandBuffer = commandQueue.makeCommandBuffer()!
        self.encoder = commandBuffer.makeComputeCommandEncoder()!
        self.captureON = false
        self.captureManager = MTLCaptureManager.shared()
        
        self.library = device.makeDefaultLibrary()!
        self.globalStates = [:]
        let functionNames = ["memcpy32", "rmsNorm32","halfToFloat", "mulVec32by16", "basicMul",
                             "repeat4x32", "mulComplex32", "dotSetScore32", "zero32", "sum_of_exps32",
                             "softmax_add32", "sumScores32", "add32", "floatToHalf", "probeExpert",
                             "basicBitonicSort", "getVal", "prepareExpertDispatch", "bucketMul",
                             "silu32", "mulScalar32x32", "memcpy16", "memcpyBig16"]
        
        
        /*["sum_of_squares32",
                             "sum_of_exps","softmax_add", "memcpy", "sumScores",
                             "dot", "setScore",  "mul_vec", "add_vec", "mul_complex",
                             "floatToHalf", "silu", "cosinePrecalc", "cosineCalc",
                             "basicBitonicSort", "probe", "getVal", "bucketMul","prepareDispatch", "zero32", "zero16",
        "cosinePrecalc16","strictDiff", "rms_norm", "dotSetScore", "silu32", "prepareDispatch32", "dotSetScore2"]*/

        for fname in functionNames {
            makeFunction(fname)
        }
    }
    
    func makeFunction(_ fname: String) {
        guard let internalFunc = library.makeFunction(name: fname) else {
            fatalError("Cannot find \"\(fname)\" in the library.")
        }
        self.globalStates[fname] = try! device.makeComputePipelineState(function: internalFunc)
    }
    
    func reEncode() {
        encoder.endEncoding()
        self.encoder = commandBuffer.makeComputeCommandEncoder()!

    }
    
    
    func eval() {
        if self.warnOfEvals {
            print("warn: EVAL")
        }
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        self.commandBuffer = commandQueue.makeCommandBuffer()!
        self.encoder = commandBuffer.makeComputeCommandEncoder()!
    }
    
    func copyBuffer(src: MTLBufferable, dst:MTLBufferable, size: Int) {
        encoder.endEncoding()

        let blitCommandEncoder = commandBuffer.makeBlitCommandEncoder()
        blitCommandEncoder?.copy(from: src.buffer,
                                 sourceOffset: src.offsetBytes,
                                 to: dst.buffer,
                                 destinationOffset: dst.offsetBytes,
                                     size: size)
        blitCommandEncoder?.endEncoding()
        self.encoder = commandBuffer.makeComputeCommandEncoder()!

    }
    
    func deploy(_ fname: String, 
                buffers: [MTLBufferable],
                ints: [Int] = [],
                float16s: [Float16] = [],
                threadCount: Int, threadCountY: Int = 1, threadCountZ: Int = 1,
                threadGroupSize tgs: [Int] = [32, 1, 1],
                justDispatch: Bool = false) {
        
        let gridSize = MTLSize(width: threadCount, height: threadCountY, depth: threadCountZ)
        let threadGroupSize = MTLSize(width: tgs[0], height: tgs[1], depth: tgs[2])

        if (!globalStates.keys.contains(fname)) {
            makeFunction(fname)
            print("warn:Compute pipeline state for \(fname) not found.")
        }
        
        let internalState = self.globalStates[fname]!
        
        
        encoder.setComputePipelineState(internalState)
        
        var idx = 0
        
        for b in buffers {
            encoder.setBuffer(b.buffer, offset: b.offsetBytes , index: idx)
            idx += 1
        }
    
        for i in ints {
            var val = i
            encoder.setBytes(&val, length: MemoryLayout<Int>.stride, index: idx)
            idx += 1
        }
        
        for i in float16s {
            var val = i
            encoder.setBytes(&val, length: MemoryLayout<Float16>.stride, index: idx)
            idx += 1
        }
            
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    }
    
    func stopCapture() {
        if (self.captureON) {
            self.captureManager.stopCapture()
            self.captureON = false
        }
    }
    
    func startCapture(cond: Bool = true) {
        if !cond { return }
        if self.captureON { return }
        
        let captureDescriptor = MTLCaptureDescriptor()
        captureDescriptor.captureObject = device
        do {
            try captureManager.startCapture(with: captureDescriptor)
        } catch {
            fatalError("error when trying to capture: \(error)")
        }
        
        self.captureON = true
        
    }
}
