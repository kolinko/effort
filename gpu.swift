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
    
    init() {
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
    
    func deploy(_ fname: String, buffers: [Bufferable], ints: [Int] = [], threadCount: Int) {
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
/*
func deploy(_ encoder: MTLComputeCommandEncoder, fname: String, buffers: [Bufferable], ints: [Int] = [], threadCount: Int) {
    if (!globalStates.keys.contains(fname)) {
            let internalFunc = library.makeFunction(name: fname)!
            globalStates[fname] = try! device.makeComputePipelineState(function: internalFunc)
            print("warn:Compute pipeline state for \(fname) not found.")
        }
    
    let internalState = globalStates[fname]!
        
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

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
}
*/
