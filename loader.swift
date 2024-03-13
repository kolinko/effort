/*
 
    Loading weights code
 
 */

import Metal
import Foundation


struct ModelData {
    let norm: Matrix
    let outputs: Matrix
    let tokEmbeddings: Matrix
    let layers: [Int: [String: Matrix]]
}

let absolutePath = "/Users/kolinko/mul_col/model/"
import Foundation

func readJson() -> [String: [Int]] {
    let fileUrl = URL(fileURLWithPath: absolutePath + "shape.json")
    let data = try! Data(contentsOf: fileUrl)
    let dictionary = try! JSONSerialization.jsonObject(with: data, options: []) as! [String: [Int]]

    return dictionary
}

func loadTokens(device: MTLDevice) -> [Vector] {
    let fileName = absolutePath + "/tokens.bin"
    let fileURL = URL(fileURLWithPath: fileName)

    let fileHandle = FileHandle(forReadingAtPath: fileURL.path)!
    let data = fileHandle.readDataToEndOfFile()

    let numTokens = 9
    let layerSize = 4096
    let expectedCount = numTokens * layerSize
    let expectedSize = expectedCount * MemoryLayout<Float32>.size

    // Assert that the data size matches the expected size
    assert(data.count == expectedSize, "Data size does not match expected size for \(fileName). Expected: \(expectedSize), Actual: \(data.count)")

    // Convert Float32 data to Float16
    let float32Pointer = data.withUnsafeBytes { $0.bindMemory(to: Float32.self) }
    var float16Data = [Float16]()
    float16Data.reserveCapacity(expectedCount)

    for i in 0..<expectedCount {
        let float32Value = float32Pointer[i]
        let float16Value = Float16(float32Value)
        float16Data.append(float16Value)
    }

    // Create MTLBuffer from Float16 data
    let buffer = device.makeBuffer(bytes: float16Data, length: float16Data.count * MemoryLayout<Float16>.size, options: .storageModeShared)!

    var tokens: [Vector] = []

    for i in 0..<numTokens {
        let offset = i * layerSize * MemoryLayout<Float16>.size
        let layerBuffer = device.makeBuffer(bytesNoCopy: buffer.contents() + offset, length: layerSize * MemoryLayout<Float16>.size, options: .storageModeShared, deallocator: nil)!
        tokens.append(Vector(shape: [layerSize], buffer: layerBuffer))
    }

    // Directly assert a specific value in the first layer (update the assertion for Float16)
    let pointer = tokens[1].buffer.contents().assumingMemoryBound(to: Float16.self)
    assert(pointer[13] == Float16(0.0132369995), "Layer value at index 13 does not match expected value in the first layer")

    assert(tokens[1][13] == Float16(0.0132369995), "Layer value at index 13 does not match expected value in the first layer")
    assert(tokens[0].test("token[0]", mul: 100, val: [0.02, -0.01, 0.01, 0.02, -0.01]))

    return tokens
}



func loadModelData(from filePath: String, device: MTLDevice) -> ModelData {
    
    let startTime = Date()
    let shapeDict = readJson()
    
    let numLayers = 31 // 31
    var layers = [Int: [String: Matrix]]()
    for i in 0...numLayers {
        layers[i] = [String: Matrix]()
        for key in ["attention.wq", "ffn_norm", "attention_norm", "attention.wv", "attention.wk", "attention.wo", "feed_forward.w1", "feed_forward.w2","feed_forward.w3"] {
            let keyName = "layers."+String(i)+"."+key
            layers[i]![key] = loadBinaryFile(named: keyName, shape: shapeDict[keyName]!, device:device)
        }
        
        for key in ["feed_forward.w1", "feed_forward.w2","feed_forward.w3"] {
            let keyName = "layers."+String(i)+"."+key
            let nShape = [shapeDict[keyName]![1], shapeDict[keyName]![0]]
            layers[i]![key+".ids"] = loadBinaryFile(named: keyName+".ids.bin", shape: nShape, device:device)
            layers[i]![key+".vals"] = loadBinaryFile(named: keyName+".vals.bin", shape: nShape, device:device)
        }
    }
    
    let model = ModelData(
        norm:loadBinaryFile(named: "norm", shape: shapeDict["norm"]!, device: device),
        outputs:loadBinaryFile(named: "output", shape: shapeDict["output"]!, device: device),
        tokEmbeddings:loadBinaryFile(named: "tok_embeddings", shape: shapeDict["tok_embeddings"]!, device: device),
        layers: layers
    )
    
    assert(model.norm[5]==1.544, "data seems loaded incorrectly?")
    let testLayer = model.layers[0]!["feed_forward.w1"]!
    assert(testLayer[4*testLayer.shape[1] + 10] == -0.02287, "wrong data on layers.0.feed_forward.w1[4][10]")
    assert(testLayer[10*testLayer.shape[1] + 4] == 0.02187, "wrong data on layers.0.feed_forward.w1[10][4]")
    assert(model.layers[0]!["feed_forward.w1.ids"]!.testInt("w1ids", val:[3260, 7938, 9263, 9670]))

    let endTime = Date()
    print("data load time \(endTime.timeIntervalSince(startTime)) seconds")

    return model
}


func loadBinaryFile(named fileName: String, shape: [Int], device: MTLDevice) -> Matrix {
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
    let buffer = device.makeBuffer(bytesNoCopy: dataPointer!, length: expectedSize, options: .storageModeShared, deallocator: nil)!

    return Matrix(shape: shape, buffer: buffer)
}

