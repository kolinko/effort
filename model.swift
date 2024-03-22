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
//    let offset_bytes: Int
    
    init(buffer: MTLBuffer, offset: Int = 0) {
//        assert((offset == 0)||(offset_bytes != 0), "You need to offset bytes if you make a regular offset")
        self.buffer = buffer
        self.offset = offset
//        self.offset_bytes = offset
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
                return bufferPointer[index+Int(offset/4)]
            }
            set(newValue) {
                let bufferPointer = self.bufferPointer
                bufferPointer[index+Int(offset/4)] = newValue
            }
        }
    /*
    subscript(index: Int) -> Float {
            get {
                let bufferPointer = self.bufferPointer
                
                return bufferPointer[index]
            }
            set(newValue) {
                let bufferPointer = self.bufferPointer
                bufferPointer[index] = newValue
            }
        }*/

    func str() -> String {
        
        var outStr = ""
        for i in 0..<32 {
            outStr += "\(self[i]); "
        }
        return outStr
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
    
    func str() -> String {
        
        var outStr = ""
        for i in 0..<32 {
            outStr += "\(self[i]); "
        }
        return outStr
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
        let result = self.test(mul: mul, val: val)
        if result {
            print("✔️ \(name)")
        } else {
            print("❌ \(name)")
        }
        return result
    }
        
    func test(mul:Int, val:[Float16]) -> Bool {
        gpu.eval()
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

func modelRunTests() {
    let v = Vector(from: [0.1, 0.22, 0.33, 0.11, -0.21, 2, -0.01, 0.02])
    assert(v.scalarAt(3)[0] == 0.11)
    v.sort()
    assert(v.test("v.sort()", mul: 100, val: [-0.21, -0.01, 0.02, 0.1, 0.11, 0.22, 0.33, 2.0]))
}

class Matrix: BufferableFloat16 {
    func asVector() -> Vector {
        return Vector(shape: [self.count()], buffer: self.buffer)
    }
    
    func scalarAt(_ row: Int, _ col: Int) -> Scalar {
        return Scalar(buffer: self.buffer, offset: row*self.cols! + col)
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


class VectorFloat: BufferableFloat {
    func cosineSimilarityTo(_ vec: Vector) -> ScalarFloat {
        let dotBuffer = ScalarFloat(value:0)
        let normABuffer = ScalarFloat(value: 0)
        let normBBuffer = ScalarFloat(value: 0)
        
        gpu.deploy("cosinePrecalc", buffers: [self, vec, dotBuffer, normABuffer, normBBuffer], threadCount: self.rows)
        gpu.deploy("cosineCalc", buffers: [dotBuffer, normABuffer, normBBuffer], threadCount: 1)
        gpu.eval()
        return dotBuffer
    }

    func asFloat16Vector() -> Vector {
        let out = Vector(shape:[self.rows])
        gpu.deploy("floatToHalf", buffers: [self, out], threadCount: self.rows)
        return out
    }
    
    
    func reshaped(newCols: Int) -> [VectorFloat] {
        // Ensure that the original layer can be evenly divided by the new dimension size
        assert(self.rows % newCols == 0, "Original layer size must be divisible by new dimension size")
        
        let newRows = self.rows / newCols
        
        var out = [VectorFloat]()
        out.reserveCapacity(newRows)
        
        for i in 0..<newRows {
            out.append(VectorFloat(shape:[newCols], buffer:self.buffer, offset: i*newCols))
        }
        
        assert(out[1][0] == self[1*newCols])
        return out
    }
}

class Vector: BufferableFloat16 {
    func scalarAt(_ row: Int) -> Scalar {
        return Scalar(buffer: self.buffer, offset: row)
    }

    
    func softmax() {
        let rms = ScalarFloat(value: 0.0)
        gpu.deploy("sum_of_exps", buffers: [self, rms], threadCount: self.rows)
        gpu.deploy("softmax_add", buffers: [self, rms], threadCount: self.rows)
    }
    
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
        
        assert(out[1][0] == self[1*newCols])
        return out
    }
    
    
    func add(by vector: Vector) {
        print(self.shape)
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
    
    
    func sort() {
        guard let logn = Int(exactly: log2(Double(self.rows))) else {
            fatalError("data.count is not a power of 2")
        }

        for p in 0..<logn {
            for q in 0..<p+1 {
                gpu.deploy("basicBitonicSort", buffers: [self], ints: [p, q], threadCount: self.rows)
            }
        }
    }
    
}

/*
 
 array funcs
 
 */




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

func sumScores(numHeads: Int, headDim:Int, scores: [Vector], xvToken: [Vector]) -> Vector {
    let outMatrix = Matrix(shape: [numHeads, headDim])
    let scoresMatrix = gpuConsolidate(vecList: scores)
    let xvTokenMatrix = gpuConsolidate(vecList: xvToken)

    let numTokens = scores[0].rows
    let numDims = numHeads*headDim
    gpu.deploy("sumScores", buffers:[scoresMatrix, xvTokenMatrix, outMatrix], ints: [numTokens], threadCount: numDims)
    return outMatrix.asVector()
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



func mul_col(vec: Vector, by weights: Matrix) -> Vector {
    assert(weights.cols == vec.rows, "Weights column count must match vec length")
    let (rows, cols) = (weights.rows, weights.cols!)

    let output = Vector(shape: [rows])

    gpu.deploy("mul_col_\(cols)", buffers:[weights, vec, output], threadCount: rows)
    
    return output
}


func silu(_ x1: VectorFloat, _ x3: VectorFloat) -> Vector {
    let out = Vector(shape:[x1.rows])
    gpu.deploy("silu", buffers: [x1, x3, out], threadCount: x1.rows)
    return out
}

func calcDispatch(v: Vector, weights: Matrix, weightBuckets: Matrix, quant: Double) -> VectorFloat {
    
    let probes = 4096 // 4096
    let o = Vector(shape: [probes])
    let wCols : Int = weights.cols!
    gpu.deploy("probe", buffers:[v, weights, o], ints:[wCols], threadCount: probes)
    
//    assert(o.test("probes", mul: 10000, val: [0.0006, 0.0012, 0.0032, 0.0005, 0.0006]))
        
    o.sort()

    let q = Int(Double(probes)*(1-quant))
    let cutoff = Scalar(value: 0)
    gpu.deploy("getVal", buffers: [o, cutoff], ints:[q], threadCount: o.rows)
    gpu.eval()
    print("cutoff", cutoff[0])
    // todo: calc the dispatch vector
    
    let dispatchStats = Matrix(shape: [weightBuckets.rows, 4]) // min, max, avg//, med
    let weightVectors = weightBuckets.asVectorList()
    var counter1 = 0
    var counter2 = 0
    var counter3 = 0
    var counter4 = 0
    let dispatch = VectorFloat(shape: [weightBuckets.rows*2])

    /* base data */
    
    let vectors = dispatch.reshaped(newCols: v.rows*2)
    var count = 0;
    for bucket_no in 0..<vectors.count {
        for row in 0..<v.rows {
            vectors[bucket_no][row*2] = Float(v[row]);// + Float16(Int.random(in:0..<);
            vectors[bucket_no][row*2+1] = Float(count); // Int.random(in:0..<vectors.count*v.rows))//
            count += 1;
        }
    }
    /* end of base data */
    
    /*
    for row in 0..<weightVectors.count {
        var min: Float16 = 99
        var max: Float16 = 0
        var sum: Float16 = 0
        for i in 0..<weightVectors[row].rows {
            let val = weightVectors[row][i]
            if abs(val) < min {
                min = abs(val)
            };
            if abs(val) > max {
                max = abs(val)
            };
            sum += abs(val);
            
        }
        
        dispatchStats[row*4+0] = min
        dispatchStats[row*4+1] = max
        let avg = sum/Float16(weightVectors[row].rows)
        dispatchStats[row*4+2] = avg
        
        let roNo = row % 4096
        let coff = cutoff[0] / abs(v[roNo])
        
        if coff > min {
            counter1 += 1
        }
        if coff < max {
            counter2 += 1
        }
        if coff > avg {
            counter3 += 1
        }
        if coff < avg {
            let counter = counter4
            dispatch[counter*2] = Float(v[roNo])
            dispatch[counter*2+1] = Float(row)
            counter4 += 1
        }
    }
    
    print("counters", counter1, counter2, counter3, counter4)
    
    
    for i in 8000..<8020 {
        print(dispatch[i*2+1])
    }
    
    */
    
    return dispatch
}




func bucketMul(v: Vector, weightBuckets: Matrix, weights: Matrix, out: VectorFloat, dispatch: VectorFloat) {
    let numBatches = 16
    let bucketSize = 16
    let numBuckets = out.rows / bucketSize // 11k/16 = ~688

    assert(weightBuckets.shape == [numBatches*v.rows, numBuckets])
    
    assert(numBuckets % 4 == 0)
    assert(dispatch.rows % 256 == 0)

    gpu.deploy("bucketMul", buffers: [weightBuckets, dispatch, out], ints: [dispatch.rows, weightBuckets.cols!], threadCount: weightBuckets.cols!, threadCountY:32)
    
}

/*
 
 Weights are in a bucket weight format:
 
 atch 1
            row 1: bucket0-15, bucket16-31, bucket32-47...
            row 2: ...
            ...


         batch 2
         ...

         total number of rows:
             num_batches * shape[0]
         total positions per row:
             batch_size
 
 */
