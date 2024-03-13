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

class Bufferable {
    let buffer: MTLBuffer
    let offset: Int
    
    init(buffer: MTLBuffer, offset: Int = 0) {
        self.buffer = buffer
        self.offset = offset
    }
}

class BufferableFloat: Bufferable {
    let bufferPointer: UnsafeMutablePointer<Float>
    let shape: [Int]
    let rows: Int
    let cols: Int?
    
    init(shape: [Int], buffer: MTLBuffer, offset: Int = 0) {
        self.rows = shape[0]
        self.cols = shape.count >= 2 ? shape[1] : nil
        self.shape = shape
        self.bufferPointer = buffer.contents().bindMemory(to: Float.self, capacity: self.shape.reduce(1, *))
        super.init(buffer: buffer, offset: offset*MemoryLayout<Float>.size)
    }
    
    convenience init(shape: [Int]) {
        let bufferSize = shape.reduce(1, *) * MemoryLayout<Float>.size
        let buffer = gpu.device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        self.init(shape: shape, buffer: buffer)
    }
    
    subscript(index: Int) -> Float {
            get {
                let bufferPointer = self.bufferPointer
                return bufferPointer[index]
            }
            set(newValue) {
                let bufferPointer = self.bufferPointer
                bufferPointer[index] = newValue
            }
        }
}

class ScalarFloat: BufferableFloat {
    
    convenience init(value: Float) {
        self.init(shape: [1])
        self[0] = value;
    }
    
}


class Scalar: BufferableFloat16 {
    
    convenience init(value: Float16) {
        self.init(shape: [1])
        self[0] = value;
    }
    
    convenience init(buffer: MTLBuffer, offset: Int = 0) {
        self.init(shape: [1], buffer: buffer, offset: offset)
    }
    
}

class BufferableFloat16 : Bufferable {
    let bufferPointer: UnsafeMutablePointer<Float16>
    let shape: [Int]
    let rows: Int
    let cols: Int?

    
    init(shape: [Int], buffer: MTLBuffer, offset: Int = 0) {
        self.rows = shape[0]
        self.cols = shape.count >= 2 ? shape[1] : nil
        self.shape = shape
        self.bufferPointer = buffer.contents().bindMemory(to: Float16.self, capacity: self.shape.reduce(1, *))
        super.init(buffer: buffer, offset: offset*MemoryLayout<Float16>.size)
    }
    
    
    convenience init(shape: [Int]) {
        let numElements = shape.reduce(1, *)
        let bufferSize = numElements * MemoryLayout<Float16>.size
        let buffer = gpu.device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        self.init(shape: shape, buffer: buffer)
    }


    convenience init(shape: [Int], with: Float16) {
        self.init(shape: shape)
        for i in 0..<self.count() {
            self[i] = with
        }
    }
    
    convenience init(from array: [Float16]) {
        assert(!array.isEmpty, "Array must not be empty")
        let length = array.count * MemoryLayout<Float16>.size
        let buffer = gpu.device.makeBuffer(bytes: array, length: length, options: .storageModeShared)!
        self.init(shape: [array.count], buffer: buffer)
    }
        
    func count() -> Int {
        return self.shape.reduce(1, *)
    }
    
    
    func getInt(index: Int) -> Int16 {
        var floatStorage: Float16 = self[index]//1.0
        var intStorage: Int16 = 0

        withUnsafePointer(to: &floatStorage) { floatPointer in
            floatPointer.withMemoryRebound(to: Int16.self, capacity: 1) { intPointer in
                intStorage = intPointer.pointee
            }
        }
        return intStorage
    }

    
    subscript(index: Int) -> Float16 {
            get {
                let bufferPointer = self.bufferPointer
                return bufferPointer[index+Int(offset/2)]
            }
            set(newValue) {
                let bufferPointer = self.bufferPointer
                bufferPointer[index+Int(offset/2)] = newValue
            }
        }
    
    
    func test(_ name: String, cond: Bool = true, mul:Int, val:[Float16]) -> Bool {
        if (!cond) {
            return true
        }
        gpu.eval()
//        return true
        let result = self.test(mul: mul, val: val)
        if result {
            print("✔️ \(name)")
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
    
    func testInt(_ name: String, val:[Int16]) -> Bool {
        let result = self.testInt(val: val)
        if result {
            print("✔️ \(name)")
        } else {
            print("❌ \(name)")
        }
        return result
    }
        
    func testInt(val:[Int16]) -> Bool {
        for i in 0..<val.count {
            if (self.getInt(index: i) != val[i]) {
                print("assert failed for values")
                for j in 0..<val.count {
                    print(self.getInt(index:j))
                }
                print("assert failed, on pos \(i), \(self.getInt(index: i)) ≠ \(val[i])")
                return false
            }
        }
        return true
    }
    
}

class Matrix: BufferableFloat16 {
    func asVector() -> Vector {
        return Vector(shape: [self.count()], buffer: self.buffer)
    }
    
    func scalarAt(_ row: Int, _ col: Int) -> Scalar {
        return Scalar(buffer: self.buffer, offset: row*self.cols! + col)
        //scores.scalarAt(headNo, t2)
    }
    
    func asVectorList() -> [Vector] {
        var out = [Vector]()
        out.reserveCapacity(self.rows)
        for i in 0..<self.rows {
            out.append(Vector(shape:[self.cols!], buffer:self.buffer, offset: i*self.cols!))
        }
        return out
    }
}

class Vector: BufferableFloat16 {
    func rmsNormed() -> Vector {
        let layer = self
        assert(layer.shape.count == 1, "Only for vectors")
        
        let output = Vector(shape: layer.shape)
        let rms = ScalarFloat(value:0.0)
        
        gpu.deploy("sum_of_squares", buffers: [layer, rms], threadCount: layer.count())
        gpu.deploy("normalize_vector", buffers: [layer, output, rms], ints: [self.count()], threadCount: layer.count())
        
        return output
    }
    
    
    func reshaped(newCols: Int) -> [Vector] {
        // Ensure that the original layer can be evenly divided by the new dimension size
        assert(self.rows % newCols == 0, "Original layer size must be divisible by new dimension size")
        
        let newRows = self.rows / newCols
        
        var out = [Vector]()
        out.reserveCapacity(newRows)
        
        for i in 0..<newRows {
            out.append(Vector(shape:[newCols], buffer:self.buffer, offset: i*newCols))
        }
        
        assert(out[3][0] == self[3*newCols])
        return out
    }
    
    
    func add(by vector: Vector) {
        assert(self.shape == vector.shape, "Shapes of both layers must match")

        gpu.deploy("add_vec", buffers:[self, vector, self], threadCount: self.rows)
    }

    func mul(by wa: Vector) {
        assert(self.shape == wa.shape)
        
        gpu.deploy("mul_vec", buffers:[self, wa, self], threadCount:self.rows)
    }
    
    
    func mul(complexArray: Vector) {
        // Ensure the layer has the correct number of elements
        let count: Int = self.shape[0] / 2
        assert(self.shape[0] == complexArray.rows, "Layer size must be twice the size of the complex array")

        gpu.deploy("mul_complex", buffers: [self, complexArray], threadCount: count)
    }
    
}

/*
 
 array funcs
 
 */

func softmax(_ layer: inout Vector) {
    let rms = ScalarFloat(value: 0.0)
    
    gpu.deploy("sum_of_exps", buffers: [layer, rms], threadCount: layer.count())
    gpu.deploy("softmax_add", buffers: [layer, rms], threadCount: layer.count())
}


/// freqs

func createFreqsCis(headDim: Int, maxSeqLen: Int) -> [Vector] {
    func logspace(start: Double, end: Double, num: Int, base: Double = 10.0) -> [Double] {
        assert(num>1)
        let step = (end - start) / Double(num)
        return (0..<num).map { pow(base, start + Double($0) * step) }
    }

    assert(headDim==128, "unusual headDim. it should work with others, but asserts/tests will fail")
    let freqs = logspace(start: 0, end: 1.0, num: headDim / 2, base: 1e-4)
    assert(freqs[2] == 0.7498942093324559)
    let heads = Matrix(shape: [2*maxSeqLen, freqs.count*2]).asVectorList()
    for i in 0..<(2 * maxSeqLen) {
        for j in 0..<freqs.count {
            let freq = freqs[j]
            let angle = Float(i) * Float(freq)
            let realPart = Float16(cos(angle))
            let imagPart = Float16(sin(angle))
            heads[i][j*2] = realPart
            heads[i][j*2+1] = imagPart
        }
    }
    assert(heads[1][2]==0.6479058)
    assert(heads[1][3]==0.7617204)
    return heads
}

func calcScores(xq_heads: [Vector], xkTokenHeads: [[Vector]]) -> [Vector] {
    let numTokens = xkTokenHeads.count
    let scores = Matrix(shape: [numHeads, numTokens])
    for t2 in 0..<numTokens {
        for headNo in 0..<numHeads {
            assert(xq_heads[headNo].rows == xkTokenHeads[t2][headNo].rows)
            let sum = ScalarFloat(value: 0)

            gpu.deploy("dot", buffers: [xq_heads[headNo], xkTokenHeads[t2][headNo], sum], threadCount:xq_heads[headNo].rows)
            gpu.deploy("setScore", buffers:[sum, scores.scalarAt(headNo, t2)], threadCount: 1)
        }
    }
    
    return scores.asVectorList()
}

func gpuConsolidate(vecList:[Vector]) -> Matrix {
    assert(vecList.count > 0)
    let out = Matrix(shape:[vecList.count, vecList[0].rows])

    for i in 0..<vecList.count {
        gpu.deploy("memcpy", buffers: [vecList[i], out.asVectorList()[i]], threadCount: vecList[i].rows)
    }

    return out
    
    
}

func sumScores(numHeads: Int, headDim:Int, scores: [Vector], xvToken: [Vector]) -> Matrix {
    let outMatrix = Matrix(shape: [numHeads, headDim])
    
    let scoresMatrix = gpuConsolidate(vecList: scores)
    let xvTokenMatrix = gpuConsolidate(vecList: xvToken)

    let numTokensX = scores[0].rows
    let numDims = numHeads*headDim
    gpu.deploy("sumScores", buffers:[scoresMatrix, xvTokenMatrix, outMatrix], ints: [numTokensX], threadCount: numDims)

    return outMatrix
}

func ffn(_ h: inout Vector, fxn: Vector, w1: Matrix, w2: Matrix, w3: Matrix) {
    let innerDim = 11008
    assert(w1.shape==[11008, 4096])
    assert(w2.shape==[4096, 11008])
    assert(w3.shape==[11008, 4096])
    assert(fxn.shape==[4096])
    
    let fx = Vector(shape: [innerDim])
    
    gpu.deploy("internal", buffers: [fxn, w1, w3, fx], threadCount: 11008)
    gpu.deploy("second", buffers: [w2, fx, h], threadCount: 4096)

}


func mul_col2(vec: Vector, by weights: Matrix) -> Vector {
    assert(weights.cols == vec.rows, "Weights column count must match vec length")
    let (rows, cols) = (weights.rows, weights.cols!)

    let output = Vector(shape: [rows])

    print(vec.shape)
    gpu.deploy("mul_col2_\(cols)", buffers:[weights, vec, output], threadCount: rows)
    
    return output
}

func mul_col(vec: Vector, by weights: Matrix) -> Vector {
    assert(weights.cols == vec.rows, "Weights column count must match vec length")
    let (rows, cols) = (weights.rows, weights.cols!)

    let output = Vector(shape: [rows])

    gpu.deploy("mul_col_\(cols)", buffers:[weights, vec, output], threadCount: rows)
    
    return output
}

/*
func mul_vm(v: Vector, layer: [String: Matrix], name: String) {
    // name e.g. feed_forward.w1
    let weights = layer[name]!
    let rowIds = layer[name+".ids"]!
    let rowVals = layer[name+".vals"]!
    assert (rowIds.cols == weights.rows)
    assert (rowIds.rows == weights.cols)
    
    print(weights.shape)
    print(rowIds.shape)
    print(rowVals.shape)

    let probes = 4096
    var o = Vector(shape: [probes])
    for i in 0..<probes {
        o[i] = abs(v[i] * weights[i*weights.cols! + i])
    }
    
    assert(o.test(mul: 10000, val: [0.0006, 0.0012, 0.0032, 0.0005, 0.0006]))
        
    sortVec(&o)
    assert(o[4095]==0.02194)
    assert(o[4094]==0.01575)

    let quant = 0.16
    let q = Int(Double(probes)*(1-quant))
    var cutoff: Float16 = o[q]

    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    let bufferSize = 11008 * MemoryLayout<Float>.stride
    let bufferX = weights.buffer.device.makeBuffer(length: bufferSize, options: .storageModeShared)!
    let bufferPointer = bufferX.contents().bindMemory(to: Float.self, capacity: 11008)

    let accumFunction = library.makeFunction(name: "accum")!
    let pipeline = try! device.makeComputePipelineState(function: accumFunction)
    
    let threadCount = v.rows
    let gridSize = MTLSize(width: threadCount, height: 1, depth: 1)
    let threadGroupSize = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
    
    for i in 0..<11008 {
        bufferPointer[i] = 0
    }
        
    encoder.setComputePipelineState(pipeline)

    encoder.setBuffer(v.buffer, offset: 0, index: 0)
    encoder.setBuffer(rowIds.buffer, offset: 0, index: 1)
    encoder.setBuffer(rowVals.buffer, offset: 0, index: 2)
    encoder.setBuffer(bufferX, offset: 0, index: 3)
    encoder.setBytes(&cutoff, length: MemoryLayout<Float16>.stride, index: 4)

    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)

    
    let startTime = Date()
    encoder.endEncoding()
    commandBuffer.commit()

    commandBuffer.waitUntilCompleted()
    //PROFILE
    print("YoloMMUL: \(1000*Date().timeIntervalSince(startTime), precision:2) ms")

    let dataPointer = bufferX.contents().assumingMemoryBound(to: Float.self)
    let dataBufferPointer = UnsafeMutableBufferPointer(start: dataPointer, count: 11008)
    let floatData = Array.init(dataBufferPointer)

    print("works?")
    print(cutoff)
    print("cutoff")

    for i in 0..<8 {
        print(rowIds.getInt(index:i))
//        print(rowVals[i])
    }

    
    for i in 0..<10 {
        print(floatData[i])
    }
    
    
    exit(0)
    
}*/

func sortVec(_ v: inout Vector) {
    // taken from https://developer.apple.com/forums/thread/674181
    //            https://github.com/tgymnich/MetalSort
    

    let device = MTLCreateSystemDefaultDevice()!
    let commandQueue = device.makeCommandQueue()!
    let library = device.makeDefaultLibrary()!
    let sortFunction = library.makeFunction(name: "basicBitonicSort")!
    let pipeline = try! device.makeComputePipelineState(function: sortFunction)


    guard let logn = Int(exactly: log2(Double(v.rows))) else {
        fatalError("data.count is not a power of 2")
    }
    
    let threadgroupsPerGrid = MTLSize(width: v.rows, height: 1, depth: 1)
    let threadsPerThreadgroup = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
    
    let commandBuffer = commandQueue.makeCommandBuffer()!

    let encoder = commandBuffer.makeComputeCommandEncoder()!
    let startTime = Date()

    for p in 0..<logn {
        for q in 0..<p+1 {
            var n1 = p
            var n2 = q

            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(v.buffer, offset: 0, index: 0)
            encoder.setBytes(&n1, length: MemoryLayout<Float>.stride, index: 1)
            encoder.setBytes(&n2, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
    }
    
    encoder.endEncoding()

    commandBuffer.commit()

    commandBuffer.waitUntilCompleted()

    print("basicSort: \(1000*Date().timeIntervalSince(startTime), precision:2) ms")

    }
