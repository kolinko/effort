//
//  main.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 24/01/2024.
//
/*
import Foundation
import Metal

let devices = MTLCopyAllDevices()
assert(!devices.isEmpty, "No Metal devices available")

// Optionally, you can choose a device based on specific criteria.
// For simplicity, let's use the first available device.
let device = devices[0]

print("loading")
let startTime3 = Date()
let modelData = loadModelData(from: "shape.json", device: device)
let endTime3 = Date()
print("data load time \(endTime3.timeIntervalSince(startTime3)) seconds")
print("Hello, World!")



let commandQueue = device.makeCommandQueue()!
let library = device.makeDefaultLibrary()!
let computeFunction = library.makeFunction(name: "matrixVectorMultiply")!

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
let numberOfRuns = 20
var commandBuffers: [MTLCommandBuffer] = []
var commandEncoders: [MTLComputeCommandEncoder] = []
let startTime2 = Date()

for _ in 1...numberOfRuns {
    if let commandBuffer = commandQueue.makeCommandBuffer(),
       let commandEncoder = commandBuffer.makeComputeCommandEncoder() {
        
        commandEncoder.setComputePipelineState(pipelineState)
        commandEncoder.setBuffer(matrixBuffer, offset: 0, index: 0)
        commandEncoder.setBuffer(vectorBuffer, offset: 0, index: 1)
        commandEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
        
        // Dispatch the compute command
        commandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)

        commandEncoder.endEncoding()

        commandBuffers.append(commandBuffer)
        commandEncoders.append(commandEncoder)
    }
}

let endTime2 = Date()
let timeInterval2 = endTime2.timeIntervalSince(startTime2)
print("Cumulative execution time for \(numberOfRuns) runs: \(timeInterval2) seconds")

let startTime = Date()

print(commandBuffers.count)
for n in 0...numberOfRuns-1 {
    let commandBuffer = commandBuffers[n]
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
}

let endTime = Date()
let timeInterval = endTime.timeIntervalSince(startTime)

print("Average execution time for \(numberOfRuns) runs: \(timeInterval) seconds")
    
guard let buffer = resultBuffer else {
    fatalError("resultBuffer is nil")
}

let data = NSData(bytesNoCopy: buffer.contents(), length: 10096 * MemoryLayout<Float>.size, freeWhenDone: false)
var resultArray = [Float](repeating: 0, count: 10096)
data.getBytes(&resultArray, length: resultArray.count * MemoryLayout<Float>.size)
*/
