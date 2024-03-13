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
    
    init() {
        self.commandBuffer = commandQueue.makeCommandBuffer()!
        self.encoder = commandBuffer.makeComputeCommandEncoder()!
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
