//
//  mps.swift
//  effort
//
//  Created 18/03/2024.
//

import Foundation

import Metal
import MetalPerformanceShaders


func basicMul(v: VectorFloat, by weights: Matrix, out result: VectorFloat) {
    assert(weights.rows == result.rows)
    assert(weights.cols == v.rows)
    
    assert(weights.cols % 16 == 0)
    mpsMul(v: v.asFloat16(), by: weights, out: result)
}


func mpsMul(v vector: Vector, by weights: Matrix, out result: VectorFloat) {
    result.zero()

    let matrixRows: Int = weights.rows// Number of rows in your matrix
    let matrixColumns: Int = weights.cols// Number of columns in your matrix (also the size of your vector)
    
    // Create MPSMatrixDescriptors for the matrix and the vector
    let matrixDescriptor = MPSMatrixDescriptor(rows: matrixRows, columns: matrixColumns, rowBytes: matrixColumns * MemoryLayout<Float16>.stride, dataType: .float16)
    
    let vectorDescriptor = MPSVectorDescriptor(length: matrixColumns, dataType: .float16)
    
    let matrix = MPSMatrix(buffer: weights.buffer, descriptor: matrixDescriptor)
    let vector = MPSVector(buffer: vector.buffer, descriptor: vectorDescriptor)
    
    // Result vector descriptor and buffer
    let resultVectorDescriptor = MPSVectorDescriptor(length: matrixRows, dataType: .float32)
    let resultVector = MPSVector(buffer: result.buffer, descriptor: resultVectorDescriptor)
    
    // Create a MPSMatrixVectorMultiplication object to perform the multiplication
    let mvmul = MPSMatrixVectorMultiplication(device: gpu.device, transpose: false, rows: matrixRows, columns: matrixColumns, alpha: 1.0, beta: 1.0)
    
    gpu.encoder.endEncoding()
    mvmul.encode(commandBuffer: gpu.commandBuffer, inputMatrix: matrix, inputVector: vector, resultVector: resultVector)
    gpu.encoder = gpu.commandBuffer.makeComputeCommandEncoder(dispatchType: gpu.dType)!
}

func mpsTopK(v: VectorFloat, topK: Int = 16)  -> VectorFloat {
    // Assuming `device` and `commandQueue` are already initialized
    // Shapes of the matrix and vector
    let topKIdxs = VectorFloat(shape:[topK])
    let topKVals = VectorFloat(shape:[topK])

    mpsTopK(v: v, outIndexes: topKIdxs, outValues: topKVals)
    
    return topKIdxs
}

func mpsTopK(v: VectorFloat, topK: Int = 16, outIndexes: VectorFloat, outValues: VectorFloat) {
    // Shapes of the matrix and vector
    assert(v.byteSize == 4)
    assert(outIndexes.byteSize == 4)
    assert(outValues.byteSize == 4)
    assert(outIndexes.rows == topK)
    assert(outValues.rows == topK)
    
    let matrixDescriptor = MPSMatrixDescriptor(rows: 1, columns: v.rows, rowBytes: v.rows * v.byteSize, dataType: .float32)
    let matrix = MPSMatrix(buffer: v.buffer, descriptor: matrixDescriptor)

    let outMatrixDescriptor = MPSMatrixDescriptor(rows: 1, columns: topK, rowBytes: topK * outIndexes.byteSize, dataType: .float32)
    let valueMatrix = MPSMatrix(buffer: outValues.buffer, descriptor: outMatrixDescriptor)
    let indexMatrix = MPSMatrix(buffer: outIndexes.buffer, descriptor: outMatrixDescriptor)

    let findTopK = MPSMatrixFindTopK(device: gpu.device, numberOfTopKValues: topK)
    
    gpu.encoder.endEncoding()
    findTopK.encode(commandBuffer: gpu.commandBuffer, inputMatrix: matrix, resultIndexMatrix: indexMatrix, resultValueMatrix: valueMatrix)
    gpu.encoder = gpu.commandBuffer.makeComputeCommandEncoder(dispatchType: gpu.dType)!
}
