import Foundation
import Metal

extension String.StringInterpolation {
    mutating func appendInterpolation(_ value: Double, precision: Int) {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.locale = Locale(identifier: "en_US_POSIX") // Use a locale with '.' as the decimal separator
        formatter.minimumFractionDigits = precision
        formatter.maximumFractionDigits = precision
        if let formattedString = formatter.string(for: value) {
            appendLiteral(formattedString)
        }
    }
}

struct Layer {
    let shape: [Int]
    let buffer: MTLBuffer
    let bufferPointer: UnsafeMutablePointer<Float16>
    let rows: Int
    let cols: Int?
    
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
        self.rows = shape[0]
        self.cols = shape.count >= 2 ? shape[1] : nil
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
//            print("✔️ \(name)")
        } else {
            print("❌ \(name)")
        }
        return result
    }
        
    func test(mul:Int, val:[Float16]) -> Bool {
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



func mul(vec: Layer, by wa: Layer) -> Layer {
    assert(vec.shape == wa.shape)
    
    var output = Layer(shape: vec.shape, device: vec.buffer.device)
    
    // Perform element-wise multiplication
    for i in 0..<vec.count() {
        output[i] = vec[i] * wa[i]
    }
    
    return output
}

func mul_col(vec: Layer, by weights: Layer) -> Layer {
    assert(weights.cols == vec.rows, "Weights column count must match vec length")
    let (rows, cols) = (weights.rows, weights.cols!)
    let startTime = Date()

    let output = Layer(shape: [rows], device: weights.buffer.device)

    
    var pipelineState : MTLComputePipelineState
    if cols == 4096 {
        pipelineState = pipelineState4096
    } else {
        assert(cols == 11008)
        pipelineState = pipelineState11008
    }
    
    
    let threadCount = rows
    let gridSize = MTLSize(width: threadCount, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: min(pipelineState.threadExecutionWidth, threadCount), height: 1, depth: 1)


    let commandBuffer = commandQueue.makeCommandBuffer()!
    let commandEncoder = commandBuffer.makeComputeCommandEncoder()!

    commandEncoder.setComputePipelineState(pipelineState)
    
    commandEncoder.setBuffer(weights.buffer, offset: 0, index: 0)
    commandEncoder.setBuffer(vec.buffer, offset: 0, index: 1)
    commandEncoder.setBuffer(output.buffer, offset: 0, index: 2)
    
    // Dispatch the compute command
    commandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)

    commandEncoder.endEncoding()
    commandBuffer.commit()

    commandBuffer.waitUntilCompleted()

    print("Mul_\(cols) total: \(1000*Date().timeIntervalSince(startTime), precision:2) ms")
    
    return output
}

