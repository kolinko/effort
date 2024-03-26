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
    
    convenience init(fromFile: String, shape: [Int]) {
        let core = loadBinaryFile(named: fromFile, shape: shape)
        let bShape = [shape[1]*16, shape[0]/16]
        let buckets = loadBinaryFile(named: fromFile+".bins.bin", shape: bShape)
        let sShape = [shape[1]*16, 4]
        let stats = loadBinaryFile(named: fromFile+".bins.stats.bin", shape: sShape)
        self.init(core: core, buckets: buckets, stats: stats)
    }
}

class Layer {
    var data = [String: Matrix]()

    private func getWeights(_ key: String) -> Weights {
        guard let core = data[key],
              let buckets = data["\(key).bins"],
              let stats = data["\(key).bins.stats"] else {
            fatalError("Invalid key or missing data for \(key)")
        }
        return Weights(core: core, buckets: buckets, stats: stats)
    }

    private func getVector(_ key: String) -> Vector {
        guard let matrix = data[key] else {
            fatalError("Matrix not found for key: \(key)")
        }
        return matrix.asVector()
    }

    var attnNorm: Vector { getVector("attention_norm") }
    var ffnNorm: Vector { getVector("ffn_norm") }

    var wo: Weights { getWeights("attention.wo") }
    var wq: Weights { getWeights("attention.wq") }
    var wk: Weights { getWeights("attention.wk") }
    var wv: Weights { getWeights("attention.wv") }

    var w1: Weights { getWeights("feed_forward.w1") }
    var w2: Weights { getWeights("feed_forward.w2") }
    var w3: Weights { getWeights("feed_forward.w3") }

    subscript(index: String) -> Matrix {
        get { data[index]! }
        set { data[index] = newValue }
    }
    
    func load(key: String, fname: String, shape: [Int]) {
        self.data[key] = loadBinaryFile(named: fname, shape: shape)
    }

}

class ModelData {
    let norm: Matrix
    let output: Weights
    let tokEmbeddings: Matrix
    let layers: [Int: Layer]
    
    init(norm: Matrix, output: Weights, tokEmbeddings: Matrix, layers: [Int : Layer]) {
        self.norm = norm
        self.output = output
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


func loadModelData(from filePath: String) -> ModelData {
    let startTime = Date()
    let shapeDict = readJson()
    
    let numLayers = 31
    var layers = [Int: Layer]()
    for i in 0...numLayers {
        layers[i] = Layer()
        for key in ["ffn_norm", "attention_norm"] {
            let elName = "layers.\(i).\(key)"
            layers[i]!.load(key:key, fname: elName, shape: shapeDict[elName]!)
        }
        
        for key in ["feed_forward.w1", "feed_forward.w2","feed_forward.w3", 
                    "attention.wv", "attention.wk", "attention.wq", "attention.wo"] {
            let elName = "layers.\(i).\(key)"
            let shape = shapeDict[elName]!
            layers[i]!.load(key: key, fname: elName, shape: shape)
            layers[i]!.load(key: key+".bins", fname: elName+".bins.bin", shape: [shape[1]*16, shape[0]/16])
            layers[i]!.load(key: key+".bins.stats", fname: elName+".bins.stats.bin", shape: [shape[1]*16, 4])
        }
    }
    
    let model = ModelData(
        norm:loadBinaryFile(named: "norm", shape: shapeDict["norm"]!),
        output: Weights(fromFile: "output", shape: shapeDict["output"]!),
        tokEmbeddings:loadBinaryFile(named: "tok_embeddings", shape: shapeDict["tok_embeddings"]!),
        layers: layers
    )
    
    assert(model.norm[5]==1.544, "data seems loaded incorrectly?")
    let testLayer = model.layers[0]!["feed_forward.w1"]
    assert(testLayer[4*testLayer.shape[1] + 10] == -0.02287, "wrong data on layers.0.feed_forward.w1[4][10]")
    assert(testLayer[10*testLayer.shape[1] + 4] == 0.02187, "wrong data on layers.0.feed_forward.w1[10][4]")

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

