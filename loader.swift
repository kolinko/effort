/*
 
    Loading weights code
 
 */

import Metal
import Foundation

class Weights {
    let core: Matrix
    let buckets: Matrix
    let stats: Matrix
    var outSize: Int { return core.rows }
    var inSize: Int { return core.cols! }
    
    init(core: Matrix, buckets:Matrix, stats:Matrix) {
        self.core = core
        self.buckets = buckets
        self.stats = stats
        assert(core.cols!*16 == buckets.rows)
        assert(core.cols!*16 == stats.rows)
    }
    
    init(elName: String, shapeDict: [String: [Int]]) {
        let shape = shapeDict[elName]!
        self.core = Matrix(fname: elName+".core.bin", shape: shape)
        self.buckets = Matrix(fname: elName+".buckets.bin", shape: [shape[1]*16, shape[0]/16])
        self.stats = Matrix(fname: elName+".bucket.stats.bin", shape: [shape[1]*16, 4])

    }
    
    convenience init(fromFile: String, shape: [Int]) {
        let core : Matrix = loadBinaryFile(named: fromFile+".core.bin", shape: shape)
        let bShape = [shape[1]*16, shape[0]/16]
        let buckets : Matrix = loadBinaryFile(named: fromFile+".buckets.bin", shape: bShape)
        let sShape = [shape[1]*16, 4]
        let stats : Matrix = loadBinaryFile(named: fromFile+".bucket.stats.bin", shape: sShape)
        self.init(core: core, buckets: buckets, stats: stats)
    }
}

class ExpertFfn {
    let w1: Weights
    let w2: Weights
    let w3: Weights
    
    init(layerName: String, shapeDict: [String: [Int]]) {
        w1 = Weights(elName: layerName+"w1", shapeDict: shapeDict)
        w2 = Weights(elName: layerName+"w2", shapeDict: shapeDict)
        w3 = Weights(elName: layerName+"w3", shapeDict: shapeDict)
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

    let attnNorm: Vector
    let ffnNorm: Vector
    let ffnGate: Matrix

    
    let wo: Weights
    let wq: Weights
    let wk: Weights
    let wv: Weights

    let experts: [ExpertFfn]

    subscript(index: String) -> Matrix {
        get { data[index]! }
        set { data[index] = newValue }
    }
        
    init(_ layerNo: Int, shapeDict: [String: [Int]]) {
        let layerName = "layers.\(layerNo)."
        
        self.ffnNorm = Vector(fname: layerName+"ffn_norm.bin", shape: shapeDict[layerName+"ffn_norm"]!)
        self.attnNorm = Vector(fname: layerName+"attention_norm.bin", shape: shapeDict[layerName+"attention_norm"]!)
        self.ffnGate = Matrix(fname: layerName+"feed_forward.gate.bin", shape: shapeDict[layerName+"feed_forward.gate"]!)
        
        self.wo = Weights(elName: layerName+"attention.wo", shapeDict: shapeDict)
        self.wk = Weights(elName: layerName+"attention.wk", shapeDict: shapeDict)
        self.wq = Weights(elName: layerName+"attention.wq", shapeDict: shapeDict)
        self.wv = Weights(elName: layerName+"attention.wv", shapeDict: shapeDict)

        self.experts = (0..<8).map { eNo in
            ExpertFfn(layerName: "layers.\(layerNo).feed_forward.experts.\(eNo).", shapeDict: shapeDict)
        }
    }

}

let absolutePath = "/Users/kolinko/mul_col/model-mixtral/"

class Model {
    let norm: Matrix
    let output: Weights
    let tokEmbeddings: Matrix
    let layers: [Int: Layer]
    
    init(from filePath: String) {
        let startTime = Date()
        let shapeDict = readJson()
        
        let numLayers = 31
        
        self.norm = loadBinaryFile(named: "norm.bin", shape: shapeDict["norm"]!)
        self.output = Weights(fromFile: "output", shape: shapeDict["output"]!)
        self.tokEmbeddings = loadBinaryFile(named: "tok_embeddings.core.bin", shape: shapeDict["tok_embeddings"]!)

        var layers = [Int: Layer]()
        for i in 0...numLayers {
            layers[i] = Layer(i, shapeDict: shapeDict)
        }
        self.layers = layers

        print("data load time \(Date().timeIntervalSince(startTime)) seconds")

    }
}


func readJson() -> [String: [Int]] {
    let fileUrl = URL(fileURLWithPath: absolutePath + "index.json")
    let data = try! Data(contentsOf: fileUrl)
    let dictionary = try! JSONSerialization.jsonObject(with: data, options: []) as! [String: [Int]]

    return dictionary
}

func loadBinaryFile(named fileName: String, shape: [Int]) -> MTLBuffer {
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
    return buffer
}

func loadBinaryFile(named fileName: String, shape: [Int]) -> Matrix {
    return Matrix(shape: shape, buffer: loadBinaryFile(named: fileName, shape: shape))
}

