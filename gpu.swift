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
        let functionNames = ["sum_of_squares", "normalize_vector",
                             "sum_of_exps","softmax_add", "memcpy", "sumScores",
                             "dot", "setScore", "internal", "second", "mul_col_4096", "mul_vec", "add_vec", "mul_complex"] // Add more function names as needed

        for fname in functionNames {
            makeFunction(fname)
        }
    }
    
    func makeFunction(_ fname: String) {
        let internalFunc = library.makeFunction(name: fname)!
        self.globalStates[fname] = try! device.makeComputePipelineState(function: internalFunc)
    }
    
    
    func eval() {
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        self.commandBuffer = commandQueue.makeCommandBuffer()!
        self.encoder = commandBuffer.makeComputeCommandEncoder()!
    }
    
    func deploy(_ fname: String, buffers: [Bufferable], ints: [Int] = [], float16s: [Float16] = [], threadCount: Int) {
        if (!globalStates.keys.contains(fname)) {
            makeFunction(fname)
            print("warn:Compute pipeline state for \(fname) not found.")
        }
        
        let internalState = self.globalStates[fname]!
            
        let gridSize = MTLSize(width: threadCount, height: 1, depth: 1)
        let threadGroupSize = MTLSize(width: internalState.threadExecutionWidth, height: 1, depth: 1)

        encoder.setComputePipelineState(internalState)

        for i in 0..<buffers.count {
            encoder.setBuffer(buffers[i].buffer, offset: buffers[i].offset, index: i)
        }

        for i in 0..<ints.count {
            var x: Int = ints[i]
            encoder.setBytes(&x, length: MemoryLayout<Int>.stride, index: i+buffers.count)
        }

        for i in 0..<float16s.count {
            var x: Float16 = float16s[i]
            encoder.setBytes(&x, length: MemoryLayout<Float16>.stride, index: i+buffers.count+ints.count)
        }

        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
    }
    
    func stopCapture(cond: Bool = true) {
        if cond && self.captureON {
            self.captureManager.stopCapture()
            self.captureON = false
        }
    }
    
    func startCapture(cond: Bool = true) {
        if !cond { return }
        
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
