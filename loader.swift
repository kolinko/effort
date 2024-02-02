/*
 
    Loading weights code
 
 */

import Metal
import Foundation


struct ModelData {
    let norm: Layer
    let outputs: Layer
    let tokEmbeddings: Layer
    let layers: [Int: [String: Layer]]
}

let absolutePath = "/Users/kolinko/mul_col/model/"
import Foundation

func readJson() -> [String: [Int]] {
    let fileUrl = URL(fileURLWithPath: absolutePath + "shape.json")
    let data = try! Data(contentsOf: fileUrl)
    let dictionary = try! JSONSerialization.jsonObject(with: data, options: []) as! [String: [Int]]

    return dictionary
}

func loadTokens(device: MTLDevice) -> [Layer] {
    let fileName = absolutePath + "/tokens.bin"
    let fileURL = URL(fileURLWithPath: fileName)

    let fileHandle = FileHandle(forReadingAtPath: fileURL.path)!
    let data = fileHandle.readDataToEndOfFile()

    let numLayers = 9
    let layerSize = 4096
    let expectedCount = numLayers * layerSize
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

    var layers: [Layer] = []

    for i in 0..<numLayers {
        let offset = i * layerSize * MemoryLayout<Float16>.size
        let layerBuffer = device.makeBuffer(bytesNoCopy: buffer.contents() + offset, length: layerSize * MemoryLayout<Float16>.size, options: .storageModeShared, deallocator: nil)!
        layers.append(Layer(shape: [layerSize], buffer: layerBuffer))
    }

    // Directly assert a specific value in the first layer (update the assertion for Float16)
    let pointer = layers[1].buffer.contents().assumingMemoryBound(to: Float16.self)
    assert(pointer[13] == Float16(0.0132369995), "Layer value at index 13 does not match expected value in the first layer")

    assert(layers[1][13] == Float16(0.0132369995), "Layer value at index 13 does not match expected value in the first layer")

    return layers
}



func loadModelData(from filePath: String, device: MTLDevice) -> ModelData {
    
    let startTime = Date()
    let shapeDict = readJson()
    /*
    let numLayers = 3 // or your actual number of layers
    var layers = [Int: [String: Layer]]()

    // Create a dispatch group to sync completion
    let dispatchGroup = DispatchGroup()

    // Concurrent queue for loading layers
    let layerQueue = DispatchQueue(label: "layerQueue", attributes: .concurrent)

    for i in 0...numLayers {
        layers[i] = [String: Layer]()

        for key in ["attention.wq", "ffn_norm", "attention_norm", "attention.wv", "attention.wk", "attention.wo", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"] {
            dispatchGroup.enter()
            layerQueue.async {
                let keyName = "layers.\(i).\(key)"
                let layer = loadBinaryFile(named: keyName, shape: shapeDict[keyName]!, device: device)

                // Synchronize access to the layers dictionary with a barrier
                layerQueue.async(flags: .barrier) {
                    layers[i]![key] = layer
                    dispatchGroup.leave()
                }
            }
        }
    }

    dispatchGroup.wait()*/
    
    let numLayers = 3 // 31
    var layers = [Int: [String: Layer]]()
    for i in 0...numLayers {
        layers[i] = [String: Layer]()
        for key in ["attention.wq", "ffn_norm", "attention_norm", "attention.wv", "attention.wk", "attention.wo", "feed_forward.w1", "feed_forward.w2","feed_forward.w3"] {
            let keyName = "layers."+String(i)+"."+key
            layers[i]![key] = loadBinaryFile(named: keyName, shape: shapeDict[keyName]!, device:device)
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

    let endTime = Date()
    print("data load time \(endTime.timeIntervalSince(startTime)) seconds")

    
    return model
}


func loadBinaryFile(named fileName: String, shape: [Int], device: MTLDevice) -> Layer {
    let fileURL = URL(fileURLWithPath: absolutePath + fileName)

    // Open the file and read its contents
    let fileHandle = FileHandle(forReadingAtPath: fileURL.path)!
    let data = fileHandle.readDataToEndOfFile()

    // Calculate the expected size based on shape and data type (Float16)
    let expectedCount = shape.reduce(1, *)
    let expectedSize = expectedCount * MemoryLayout<Float16>.size

    // Assert that the data size matches the expected size
    assert(data.count == expectedSize, "Data size does not match expected size for \(fileName). Expected: \(expectedSize), Actual: \(data.count)")

    // Create a MTLBuffer directly from the data
    let buffer = data.withUnsafeBytes { pointer -> MTLBuffer in
        guard let buffer = device.makeBuffer(bytes: pointer.baseAddress!, length: data.count, options: .storageModeShared) else {
            fatalError("Cannot create buffer for \(fileName)")
        }
        return buffer
    }

    return Layer(shape: shape, buffer: buffer)
}

