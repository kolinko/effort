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
    let dType = MTLDispatchType.serial
    
    let fence : MTLFence
    
    init() {
        let devices = MTLCopyAllDevices()
        assert(!devices.isEmpty, "No Metal devices available")
        self.device = devices[0]
        self.commandQueue = device.makeCommandQueue()!
        self.commandBuffer = commandQueue.makeCommandBuffer()!
        self.encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: dType)!
        self.captureON = false
        self.captureManager = MTLCaptureManager.shared()
        self.fence = device.makeFence()!

        self.library = device.makeDefaultLibrary()!
        self.globalStates = [:]
        let functionNames = ["memcpy32", "rmsNorm32","halfToFloat", "mulVec32by16", "basicMul",
                             "repeat4x32", "mulComplex32_mx", "dotSetScore32", "zero32", "sum_of_exps32",
                             "softmax_add32", "sumScores32", "add32", "floatToHalf", 
                             "basicBitonicSort", "getVal", "prepareExpertDispatch", "bucketMul",
                             "silu32", "mulScalar32x32", "memcpy16", "memcpyBig16", "touch",
                             "dotSetScore2", "findCutoff", "roundUp", "setVal", "bucketMul3",
                            "cosinePrecalc32", "cosineCalc32"]
        
        
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
        self.encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: dType)!
    }

    func reEncodeConcurrent() {
        encoder.endEncoding()
        self.encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent)!
    }

    func startEncoding() {
        self.encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: dType)!
    }
    func stopEncoding() {
        encoder.endEncoding()
    }
    
    //
    //func timeIt(repeats: Int = 10000, withCapture: Bool = false, _ closure: (Int) -> Void) {

    func concurrent(_ closures: [() -> Void]) {
        for c in closures {
            gpu.encoder.updateFence(fence)
            stopEncoding()
            self.encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent)!
            gpu.encoder.waitForFence(fence)
            c()
        }
        gpu.encoder.updateFence(fence)
        stopEncoding()
        startEncoding()
        gpu.encoder.waitForFence(fence)
    }
        
    func endConcurrent() {
       stopEncoding()
       startEncoding()
       gpu.encoder.waitForFence(fence)
    }
    
    func eval(noWarn: Bool = false) {
        if self.warnOfEvals && !noWarn {
            print("warn: EVAL")
        }
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        self.commandBuffer = commandQueue.makeCommandBuffer()!
        self.encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: dType)!
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
        self.encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: dType)!

    }
    
    func deploy(_ fname: String,
                buffers: [MTLBufferable],
                ints: [Int] = [],
                float16s: [Float16] = [],
                floats: [Float] = [],

                threadCount: Int,
                threadGroupSize tgs: [Int] = [32, 1, 1]) {
        deploy(fname, buffers: buffers, ints: ints, float16s: float16s, floats: floats, threadCount: [threadCount], threadGroupSize: tgs)
    }
    
    func deploy(_ fname: String,
                buffers: [MTLBufferable],
                ints: [Int] = [],
                float16s: [Float16] = [],
                floats: [Float] = [],

                threadCount: [Int],
                threadGroupSize tgs: [Int] = [32, 1, 1]) {
        
        let gridSize = MTLSize(width: threadCount[0],
                               height: threadCount.count > 1 ? threadCount[1] : 1,
                               depth: threadCount.count > 2 ? threadCount[2] : 1)
        let threadGroupSize = MTLSize(width: tgs[0], height: tgs[1], depth: tgs[2])

        if (!globalStates.keys.contains(fname)) {
            makeFunction(fname)
         //   print("warn:Compute pipeline state for \(fname) not found.")
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
        
        for i in floats {
            var val = i
            encoder.setBytes(&val, length: MemoryLayout<Float32>.stride, index: idx)
            idx += 1
        }
            
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    }
    
    func stopCapture() {
        if (self.captureON) {
            gpu.eval()
            self.captureManager.stopCapture()
            self.captureON = false
        }
    }
    
    func startCapture(cond: Bool = true) {
        if !cond { return }
        if self.captureON { return }
        gpu.eval()

        let captureDescriptor = MTLCaptureDescriptor()
        captureDescriptor.captureObject = device
        do {
            try captureManager.startCapture(with: captureDescriptor)
        } catch {
            fatalError("error when trying to capture: \(error)")
        }
        gpu.eval()
        self.captureON = true
        
    }
}
