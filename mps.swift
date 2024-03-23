//
//  mps.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 18/03/2024.
//

import Foundation

import Metal
import MetalPerformanceShaders

func mpsMul(vector: Vector, weights: Matrix, result: Vector) {
    // Assuming `device` and `commandQueue` are already initialized
    // Shapes of the matrix and vector
    let matrixRows: Int = weights.rows// Number of rows in your matrix
    let matrixColumns: Int = weights.cols!// Number of columns in your matrix (also the size of your vector)
    
    // Create MPSMatrixDescriptors for the matrix and the vector
    let matrixDescriptor = MPSMatrixDescriptor(rows: matrixRows, columns: matrixColumns, rowBytes: matrixColumns * MemoryLayout<Float16>.stride, dataType: .float16)
    
    let vectorDescriptor = MPSVectorDescriptor(length: matrixColumns, dataType: .float16)
    
    // Initialize MPSMatrix and MPSVector objects
    let matrix = MPSMatrix(buffer: weights.buffer, descriptor: matrixDescriptor)
    let vector = MPSVector(buffer: vector.buffer, descriptor: vectorDescriptor)
    
    // Result vector descriptor and buffer
    let resultVectorDescriptor = MPSVectorDescriptor(length: matrixRows, dataType: .float16)
    let resultBuffer = result.buffer
    
    let resultVector = MPSVector(buffer: resultBuffer, descriptor: resultVectorDescriptor)
    
    // Create a MPSMatrixVectorMultiplication object to perform the multiplication
    let matrixVectorMultiplication = MPSMatrixVectorMultiplication(device: gpu.device, transpose: false, rows: matrixRows, columns: matrixColumns, alpha: 1.0, beta: 1.0)
    
    // Prepare a command buffer and encode the matrix-vector multiplication
    gpu.encoder.endEncoding()

    matrixVectorMultiplication.encode(commandBuffer: gpu.commandBuffer, inputMatrix: matrix, inputVector: vector, resultVector: resultVector)
    gpu.encoder = gpu.commandBuffer.makeComputeCommandEncoder()!
}

func mpsTopK(v: Vector, result: Scalar)  -> MTLBuffer {
    let topK = 16
    // Assuming `device` and `commandQueue` are already initialized
    // Shapes of the matrix and vector
    let rowCount: Int = v.rows// Number of rows in your matrix
    
    // Create MPSMatrixDescriptors for the matrix and the vector
    let matrixDescriptor = MPSMatrixDescriptor(rows: 1, columns: rowCount, rowBytes: rowCount * MemoryLayout<Float16>.stride, dataType: .float16)
    let matrix = MPSMatrix(buffer: v.buffer, descriptor: matrixDescriptor)

    let outMatrixDescriptor = MPSMatrixDescriptor(rows: 1, columns: topK, rowBytes: topK * MemoryLayout<Float16>.stride, dataType: .float16)

    let topKValueBuffer = gpu.device.makeBuffer(length: topK * MemoryLayout<Float16>.size, options: .storageModeShared)!
    let topKIndexBuffer = gpu.device.makeBuffer(length: topK * MemoryLayout<UInt32>.size, options: .storageModeShared)!
    let valueMatrix = MPSMatrix(buffer: topKValueBuffer, descriptor: outMatrixDescriptor)
    let indexMatrix = MPSMatrix(buffer: topKIndexBuffer, descriptor: outMatrixDescriptor)

    
    // Create a MPSMatrixVectorMultiplication object to perform the multiplication
    let findTopK = MPSMatrixFindTopK(device: gpu.device, numberOfTopKValues: topK)
    
    // Prepare a command buffer and encode the matrix-vector multiplication
    gpu.encoder.endEncoding()

    findTopK.encode(commandBuffer: gpu.commandBuffer, inputMatrix: matrix, resultIndexMatrix: indexMatrix, resultValueMatrix: valueMatrix)
/*    matrixVectorMultiplication.encode(commandBuffer: gpu.commandBuffer, inputMatrix: matrix, inputVector: vector, resultVector: resultVector)*/
    gpu.encoder = gpu.commandBuffer.makeComputeCommandEncoder()!
    return topKValueBuffer


}
