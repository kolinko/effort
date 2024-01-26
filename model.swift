import Foundation

struct Layer {
    let shape: [Int]
    let data: [Float16]
}

struct ModelData {
    let norm: Layer
    let outputs: Layer
    let tokEmbeddings: Layer
    let layers: [Int: [String: Layer]]
}

let absolutePath = "/Users/kolinko/mul_col/model/"
import Foundation

func readJson() -> [String: [Int]] {
    // Get url for file
    let fileUrl = URL(fileURLWithPath: absolutePath + "shape.json")
    let data = try! Data(contentsOf: fileUrl)

    let dictionary = try! JSONSerialization.jsonObject(with: data, options: []) as! [String: [Int]]

    return dictionary
}


func loadModelData(from filePath: String) -> ModelData {
    let shapeDict = readJson()
    
    let numLayers = 3 // 31
    var layers = [Int: [String: Layer]]()
    for i in 0...numLayers {
        layers[i] = [String: Layer]()
        for key in ["attention.wq", "ffn_norm", "attention_norm", "attention.wv", "attention.wk", "attention.wo", "feed_forward.w1", "feed_forward.w2","feed_forward.w3"] {
            let keyName = "layers."+String(i)+"."+key
            layers[i]![key] = loadBinaryFile(named: keyName, shape: shapeDict[keyName]!)
        }
    }
    
    let model = ModelData(
            norm:loadBinaryFile(named: "norm", shape: shapeDict["norm"]!),
            outputs:loadBinaryFile(named: "output", shape: shapeDict["output"]!),
            tokEmbeddings:loadBinaryFile(named: "tok_embeddings", shape: shapeDict["tok_embeddings"]!),
            layers: layers
    )
    
    assert(model.norm.data[5]==1.544, "data seems loaded incorrectly?")
    let testLayer = model.layers[0]!["feed_forward.w1"]!
    assert(testLayer.data[4*testLayer.shape[1] + 10] == -0.02287, "wrong data on layers.0.feed_forward.w1[4][10]")
    assert(testLayer.data[10*testLayer.shape[1] + 4] == 0.02187, "wrong data on layers.0.feed_forward.w1[10][4]")

    return model
}


func loadBinaryFile(named fileName: String, shape: [Int]) -> Layer {
    let fileURL = URL(fileURLWithPath: absolutePath + fileName)

    let fileHandle = FileHandle(forReadingAtPath: fileURL.path)!
    
    let data = fileHandle.readDataToEndOfFile()
    let count = data.count / MemoryLayout<Float16>.size
    let pointer = data.withUnsafeBytes {
        $0.bindMemory(to: Float16.self).baseAddress!
    }
    
    let flatArray = Array(UnsafeBufferPointer(start: pointer, count: count))

    switch shape.count {
    case 1:
        // One-dimensional shape, return a flat array
        assert(count == shape[0], "Data size does not match shape")
        return Layer(shape: shape, data: flatArray)
    case 2:
        // Two-dimensional shape, return a flat array representing 2D data
        assert(count == shape[0] * shape[1], "Data size does not match shape in \(fileName). \(shape) vs \(count) vs \(data.count)")
        return Layer(shape: shape, data: flatArray)
    default:
        fatalError("Unsupported shape dimensionality")
    }
}
