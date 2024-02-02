import Foundation
import Metal

struct Layer {
    let shape: [Int]
    let buffer: MTLBuffer
    let bufferPointer: UnsafeMutablePointer<Float16>
    
    init(shape: [Int], device: MTLDevice) {
        let numElements = shape.reduce(1, *)
        let bufferSize = numElements * MemoryLayout<Float16>.size
        let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        self.init(shape: shape, buffer: buffer)
    }

    init(shape: [Int], with: Float16, device: MTLDevice) {
        self.init(shape: shape, device: device)
        for i in 0..<self.count() {
            self[i] = with
        }
    }
    
    init(shape: [Int], buffer: MTLBuffer) {
        self.shape = shape
        self.buffer = buffer
        self.bufferPointer = buffer.contents().bindMemory(to: Float16.self, capacity: self.shape.reduce(1, *))
    }
    
    init(from array: [Float16], using device: MTLDevice) {
        assert(!array.isEmpty, "Array must not be empty")
        let length = array.count * MemoryLayout<Float16>.size
        let buffer = device.makeBuffer(bytes: array, length: length, options: .storageModeShared)!
        self.init(shape: [array.count], buffer: buffer)
    }
        
    func count() -> Int {
        return self.shape.reduce(1, *)
    }
    
    func rmsNorm() -> Layer {
        let layer = self
        assert(layer.shape.count == 1, "Only for vectors")
        
        // Calculate the mean of the squares of the elements
        var sum: Float32 = 0.0
        for i in 0..<layer.count() {
            sum += pow(Float32(layer[i]), 2)
        }
        let mean = sum / Float32(layer.count())

        // Calculate the square root of the mean
        let sqrtMean = sqrt(mean + 1e-6)

        var output = Layer(shape: layer.shape, device: layer.buffer.device)

        // Normalize each element and store in the new buffer
        for i in 0..<layer.count() {
            output[i] = Float16(Float32(layer[i]) / sqrtMean)
        }

        return output
    }
    
    subscript(index: Int) -> Float16 {
            get {
                let bufferPointer = self.bufferPointer
                return bufferPointer[index]
            }
            set(newValue) {
                let bufferPointer = self.bufferPointer
                bufferPointer[index] = newValue
            }
        }
    
    func test(_ name: String, mul:Int, val:[Float16]) -> Bool {
        let result = self.test(mul: mul, val: val)
        if result {
            print("✔️ \(name)")
        } else {
            print("❌ \(name)")
        }
        return result
    }
        
    func test(mul:Int, val:[Float16]) -> Bool {
        // tests if the buffer is equal to values with a given accuracy
        
        for i in 0..<val.count {
            if round(self[i]*Float16(mul)) != round(val[i]*Float16(mul)) {
                print("assert failed for values")
                for j in 0..<val.count {
                    print(self[j])
                }
                print("assert failed, on pos \(i), \(self[i]) ≠ \(val[i])")
                return false
            }
        }
        return true
    }
    
    
}

/*
 
 array funcs
 
 */


func makeArray<T>(dims: [Int], value: T) -> Any {
    guard !dims.isEmpty else { return value }
    return Array(repeating: makeArray(dims: Array(dims.dropFirst()), value: value), count: dims.first!)
}


func softmax(_ array: inout [Float16]) {
    // Compute exponentials and sum them up
    let exps = array.map { Float16(exp(Float($0))) }
    let sumExps = exps.reduce(Float16(0.0), +)

    // Normalize each element
    for i in array.indices {
        array[i] = exps[i] / sumExps
    }
}

func dot(_ vec1: Layer, _ vec2: Layer) -> Float16 {
    assert(vec1.count() == vec2.count(), "Vectors must be of the same length")
    
    var sum: Float16 = 0.0
    for i in 0..<vec1.count() {
        sum += vec1[i] * vec2[i]
    }
    return sum
}


/// freqs

func createFreqsCis(headDim: Int, maxSeqLen: Int) -> [[(Float, Float)]] {
    func logspace(start: Double, end: Double, num: Int, base: Double = 10.0) -> [Double] {
        assert(num>1)
        let step = (end - start) / Double(num)
        return (0..<num).map { pow(base, start + Double($0) * step) }
    }

    assert(headDim==128, "unusual headDim. it should work with others, but asserts/tests will fail")
    let freqs = logspace(start: 0, end: 1.0, num: headDim / 2, base: 1e-4)
    assert(freqs[2] == 0.7498942093324559)
    let def: (Float, Float) = (0.0, 0.0)
    var heads = makeArray(dims: [2*maxSeqLen, freqs.count], value:def) as! [[(Float, Float)]]
    for i in 0..<(2 * maxSeqLen) {
        for j in 0..<freqs.count {
            let freq = freqs[j]
            let angle = Float(i) * Float(freq)
            let realPart = cos(angle)
            let imagPart = sin(angle)
            heads[i][j]=(realPart, imagPart)
        }
    }
    assert(heads[1][1]==((0.6479058, 0.7617204)))
    return heads
}

func reshape(vec: Layer, newDimSize: Int) -> [Layer] {
    // Ensure that the original layer can be evenly divided by the new dimension size
    assert(vec.shape[0] % newDimSize == 0, "Original layer size must be divisible by new dimension size")

    let numNewLayers = vec.shape[0] / newDimSize
    let vecBufferPointer = vec.buffer.contents().bindMemory(to: Float16.self, capacity: vec.shape[0])
    let device = vec.buffer.device

    var newLayers: [Layer] = []

    for i in 0..<numNewLayers {
        let newBuffer = device.makeBuffer(length: newDimSize * MemoryLayout<Float16>.size, options: .storageModeShared)!
        let newBufferPointer = newBuffer.contents().bindMemory(to: Float16.self, capacity: newDimSize)
        memcpy(newBufferPointer, vecBufferPointer + i * newDimSize, newDimSize * MemoryLayout<Float16>.size)
        newLayers.append(Layer(shape: [newDimSize], buffer: newBuffer))
    }

    assert(newLayers[3][0] == vec[3*newDimSize])
    
    return newLayers
}

func mul(layer: Layer, complexArray: [(Float, Float)]) -> Layer {
    // Ensure the layer has the correct number of elements
    
    func multiplyComplex(_ num1: (Float, Float), _ num2: (Float, Float)) -> (Float, Float) {
        let (a, b) = num1
        let (c, d) = num2
        return (a * c - b * d, a * d + b * c)
    }
    
    assert(layer.shape[0] == complexArray.count * 2, "Layer size must be twice the size of the complex array")

    let count = layer.shape[0] / 2
    let layerBufferPointer = layer.buffer.contents().bindMemory(to: Float.self, capacity: layer.shape[0])

    let device = layer.buffer.device
    let resultBuffer = device.makeBuffer(length: layer.shape[0] * MemoryLayout<Float>.size, options: .storageModeShared)!
    let resultBufferPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: layer.shape[0])

    for i in 0..<count {
        let complexNum = (layerBufferPointer[2 * i], layerBufferPointer[2 * i + 1])
        let result = multiplyComplex(complexNum, complexArray[i])
        resultBufferPointer[2 * i] = result.0     // Real part
        resultBufferPointer[2 * i + 1] = result.1 // Imaginary part
    }

    return Layer(shape: [128], buffer: resultBuffer)
}

func add(dest: inout Layer, by vector: Layer) {
    assert(dest.shape == vector.shape, "Shapes of both layers must match")

    for i in 0..<dest.count() {
        dest[i] += vector[i]
    }
}

func mul_row(vec: Layer, by weights:Layer) -> Layer {
    // Validate shapes
    assert(weights.shape[1] == vec.shape[0], "Weights row count must match vec length")

    let rows = weights.shape[0]
    let cols = weights.shape[1]
    
    var output = Layer(shape: [rows], with: 0, device: weights.buffer.device)
    
    // Perform the matrix-vector multiplication
    for i in 0..<rows {
        var sum = Float16()
        for j in 0..<cols {
            sum += vec[j] * weights[i * cols + j]
        }
        output[i] = sum
    }
    
    return output
}

func mul_col(vec: Layer, by weights: Layer) -> Layer {
    // Validate shapes
    assert(weights.shape[1] == vec.shape[0], "Weights column count must match vec length")

    let rows = weights.shape[0]
    let cols = weights.shape[1]

    // Prepare the output buffer
    var output = Layer(shape: [rows], with: 0, device: weights.buffer.device)

    // Perform the matrix-vector multiplication
    for i in 0..<rows {
        for j in 0..<cols {
            output[i] += vec[j] * weights[i * cols + j]
        }
    }

    return output
}


func mul(vec: Layer, by wa: Layer) -> Layer {
    assert(vec.shape == wa.shape)
    
    var output = Layer(shape: vec.shape, device: vec.buffer.device)
    
    // Perform element-wise multiplication
    for i in 0..<vec.count() {
        output[i] = vec[i] * wa[i]
    }
    
    return output
}

/*
 
    Loading weights code
 
 */

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

