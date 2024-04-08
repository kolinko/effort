//
//  converterX.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 08/04/2024.
//

import Foundation
/*

func goConvert() {
    let tSaver = TensorSaver(path: "./model-mixtral", model: "corefp16")

    for (fname, shape) in names {
//        if fname.contains("layers.1.") {//} && !fname.contains("core.bin"){
        if fname.contains("core.bin") {
            print(fname, shape)
            convertBinaryFile(named: fname, shape: shape, tSaver: tSaver)
        }
    }
    
    tSaver.save()
    
//    for ()
    
}
*/

/*
 
 func prepConvertBinaryFile(named fileName: String, shape: [Int], tSaver: TensorSaver) {
     print("(\"\(fileName)\", \(shape))," )
 }

 func convertBinaryFile(named fileName: String, shape: [Int], tSaver: TensorSaver) {
 /*    print("(\"\(fileName)\", \(shape))," )
     return*/
     let fileURL = URL(fileURLWithPath: absolutePath + fileName)

     // Calculate the expected size
     let expectedCount = shape.reduce(1, *)
     let expectedSize = expectedCount * MemoryLayout<Float16>.size

     // Memory map the file
     let fileDescriptor = open(fileURL.path, O_RDONLY)
     precondition(fileDescriptor != -1, "Cannot open file \(fileName).")

     let dataPointer = mmap(nil, expectedSize, PROT_READ, MAP_PRIVATE, fileDescriptor, 0)
     precondition(dataPointer != MAP_FAILED, "Memory mapping of \(fileName) failed.")

     // Create MTLBuffer from the memory-mapped data
     let buffer = gpu.device.makeBuffer(bytesNoCopy: dataPointer!, length: expectedSize, options: .storageModeShared, deallocator: nil)!
     
     var o : MTLBufferable = Vector(shape:[1])
     
     if shape.count == 1 {
         o = Vector(shape: shape, buffer: buffer)
     } else if shape.count == 2 {
         o = Matrix(shape: shape, buffer: buffer)
     } else {
         assert(false)
     }
     
     print("converting", fileName, shape)
     var layNo : Int? = extractNumber(from: fileName)
     if layNo == nil {
        layNo = 0
     }
     
     assert((o as! Bufferable<Float16>).shape != [1])
     tSaver[layNo!][fileName] = o
 //    print(layNo!)
     
 //    close(fileDescriptor)
 }


 */
