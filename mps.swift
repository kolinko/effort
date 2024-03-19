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
//    gpu.commandBuffer.commit()

    matrixVectorMultiplication.encode(commandBuffer: gpu.commandBuffer, inputMatrix: matrix, inputVector: vector, resultVector: resultVector)
    gpu.encoder = gpu.commandBuffer.makeComputeCommandEncoder()!

//    gpu.encoder
//    gpu.
//    gpu.encoder.endEncoding()

}
