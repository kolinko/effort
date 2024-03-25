/*
 
    Loading weights code
 
 */

import Metal
import Foundation

class Weights {
    let core: Matrix
    let buckets: Matrix
    let stats: Matrix
    var outSize: Int {
        return core.rows
    }
    var inSize: Int {
        return core.cols!
    }
    
    init(core: Matrix, buckets:Matrix, stats:Matrix) {
        self.core = core
        self.buckets = buckets
        self.stats = stats
        assert(core.cols!*16 == buckets.rows)
        assert(core.cols!*16 == stats.rows)
    }
    
    convenience init(fromFile: String) {
//        loadBinaryFile(named: "output", shape: shapeDict["output"]!),
        
    }
}
class Layer {
    var data = [String: Matrix]()

    private func getWeights(for key: String) -> Weights {
        guard let core = data[key],
              let buckets = data["\(key).bins"],
              let stats = data["\(key).bins.stats"] else {
            fatalError("Invalid key or missing data for \(key)")
        }
        return Weights(core: core, buckets: buckets, stats: stats)
    }

    private func getVector(for key: String) -> Vector {
        guard let matrix = data[key] else {
            fatalError("Matrix not found for key: \(key)")
        }
        return matrix.asVector()
    }

    var attnNorm: Vector { getVector(for: "attention_norm") }
    var ffnNorm: Vector { getVector(for: "ffn_norm") }

    var wo: Weights { getWeights(for: "attention.wo") }
    var wq: Weights { getWeights(for: "attention.wq") }
    var wk: Weights { getWeights(for: "attention.wk") }
    var wv: Weights { getWeights(for: "attention.wv") }

    var w1: Weights { getWeights(for: "feed_forward.w1") }
    var w2: Weights { getWeights(for: "feed_forward.w2") }
    var w3: Weights { getWeights(for: "feed_forward.w3") }

    subscript(index: String) -> Matrix {
        get { data[index]! }
        set { data[index] = newValue }
    }
}

class ModelData {
    let norm: Matrix
    let outputs: Matrix
    let tokEmbeddings: Matrix
    let layers: [Int: Layer]
    
    init(norm: Matrix, outputs: Matrix, tokEmbeddings: Matrix, layers: [Int : Layer]) {
        self.norm = norm
        self.outputs = outputs
        self.tokEmbeddings = tokEmbeddings
        self.layers = layers
    }
}

let absolutePath = "/Users/kolinko/mul_col/model/"
import Foundation

func readJson() -> [String: [Int]] {
    let fileUrl = URL(fileURLWithPath: absolutePath + "shape.json")
    let data = try! Data(contentsOf: fileUrl)
    let dictionary = try! JSONSerialization.jsonObject(with: data, options: []) as! [String: [Int]]

    return dictionary
}

func loadTokens() -> [Vector] {
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
    let buffer = gpu.device.makeBuffer(bytes: float16Data, length: float16Data.count * MemoryLayout<Float16>.size, options: .storageModeShared)!

    var tokens: [Vector] = []

    for i in 0..<numTokens {
        let offset = i * layerSize * MemoryLayout<Float16>.size
        let layerBuffer = gpu.device.makeBuffer(bytesNoCopy: buffer.contents() + offset, length: layerSize * MemoryLayout<Float16>.size, options: .storageModeShared, deallocator: nil)!
        tokens.append(Vector(shape: [layerSize], buffer: layerBuffer))
    }

    // Directly assert a specific value in the first layer (update the assertion for Float16)
    let pointer = tokens[1].buffer.contents().assumingMemoryBound(to: Float16.self)
    assert(pointer[13] == Float16(0.0132369995), "Layer value at index 13 does not match expected value in the first layer")

    assert(tokens[1][13] == Float16(0.0132369995), "Layer value at index 13 does not match expected value in the first layer")
    assert(tokens[0].test("token[0]", mul: 100, val: [0.02, -0.01, 0.01, 0.02, -0.01]))

    return tokens
}



func loadModelData(from filePath: String) -> ModelData {
    
    let startTime = Date()
    let shapeDict = readJson()
    
    let numLayers = 31
    var layers = [Int: Layer]()
    for i in 0...numLayers {
        layers[i] = Layer()
        for key in ["ffn_norm", "attention_norm"] {
            let keyName = "layers."+String(i)+"."+key
            layers[i]!.data[key] = loadBinaryFile(named: keyName, shape: shapeDict[keyName]!)
        }
        
        for key in ["feed_forward.w1", "feed_forward.w2","feed_forward.w3", "attention.wv", "attention.wk", "attention.wq",
        "attention.wo"] {
            let keyName = "layers."+String(i)+"."+key
            layers[i]!.data[key] = loadBinaryFile(named: keyName, shape: shapeDict[keyName]!)
            let nShape = [shapeDict[keyName]![1]*16, shapeDict[keyName]![0]/16]
            layers[i]![key+".bins"] = loadBinaryFile(named: keyName+".bins.bin", shape: nShape)
            let dShape = [shapeDict[keyName]![1]*16, 4]
            layers[i]![key+".bins.stats"] = loadBinaryFile(named: keyName+".bins.stats.bin", shape: dShape)
        }
    }
    
    let model = ModelData(
        norm:loadBinaryFile(named: "norm", shape: shapeDict["norm"]!),
        outputs:Weights.fromFile("output"),
        tokEmbeddings:loadBinaryFile(named: "tok_embeddings", shape: shapeDict["tok_embeddings"]!),
        layers: layers
    )
    
    assert(model.norm[5]==1.544, "data seems loaded incorrectly?")
    let testLayer = model.layers[0]!["feed_forward.w1"]
    assert(testLayer[4*testLayer.shape[1] + 10] == -0.02287, "wrong data on layers.0.feed_forward.w1[4][10]")
    assert(testLayer[10*testLayer.shape[1] + 4] == 0.02187, "wrong data on layers.0.feed_forward.w1[10][4]")
//    assert(model.layers[0]!["feed_forward.w1.ids"]!.testInt("w1ids", val:[3260, 7938, 9263, 9670]))

    let endTime = Date()
    print("data load time \(endTime.timeIntervalSince(startTime)) seconds")

    return model
}


func loadBinaryFile(named fileName: String, shape: [Int]) -> Matrix {
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

    return Matrix(shape: shape, buffer: buffer)
}

