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
                return bufferPointer[index+Int(offset/4)]
            }
            set(newValue) {
                let bufferPointer = self.bufferPointer
                bufferPointer[index+Int(offset/4)] = newValue
            }
        }

    func str(count: Int = 10) -> String {
        let _count = count<=self.rows ? count : self.rows
        var outStr = ""
        for i in 0..<_count {
            outStr += "\(self[i]); "
        }
        return outStr
    }
    
    func zero() {
        gpu.deploy("zero32", buffers: [self], threadCount: shape.reduce(1, *))
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
    
    func zero() {
        gpu.deploy("zeroVec", buffers: [self], threadCount: self.count())
    }
    
    func str(count: Int = 10) -> String {
        let _count = count<self.rows ? count : self.rows
        var outStr = ""
        for i in 0..<_count {
            outStr += "\(self[i]); "
        }
        return outStr
    }
    
    func count() -> Int {
        return self.shape.reduce(1, *)
    }
    
    
    func getInt(index: Int) -> Int16 {
        var floatStorage: Float16 = self[index]
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
            if val[i] != 666 {
                if round(self[i]*Float16(mul)) != round(val[i]*Float16(mul)) {
                    print("assert failed for values")
                    for j in 0..<val.count {
                        print(self[j])
                    }
                    print("assert failed, on pos \(i), \(self[i]) ≠ \(val[i])")
                    return false
                }
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

class ScalarFloat: BufferableFloat {
    
    convenience init(value: Float) {
        self.init(shape: [1])
        self[0] = value;
    }
    
    
    override func zero() {
        gpu.deploy("zero32", buffers: [self], threadCount: 1)
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

class DynaVectorFloat: VectorFloat {
    let size: ScalarFloat = ScalarFloat(value:0)
}

let _normABuffer = ScalarFloat(value: 0)
let _normBBuffer = ScalarFloat(value: 0)


class VectorFloat: BufferableFloat {
    func cosineSimilarityTo(_ vec: Vector) -> ScalarFloat {
        let dotBuffer = ScalarFloat(value:0)
        _normABuffer.zero()
        _normBBuffer.zero()
        gpu.deploy("cosinePrecalc", buffers: [self, vec, dotBuffer, _normABuffer, _normBBuffer], threadCount: self.rows)
        gpu.deploy("cosineCalc", buffers: [dotBuffer, _normABuffer, _normBBuffer], threadCount: 1)
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


let _rms = ScalarFloat(value: 0.0)

class Vector: BufferableFloat16 {
    func strictCompareTo(_ vec: Vector) -> Bool {
        _normABuffer.zero()
        gpu.deploy("strictDiff", buffers: [self, vec, _normABuffer], threadCount: self.rows)
        gpu.eval()
        return _normABuffer[0] == 0
    }
    func strictCompareTo2(_ vec: Vector) -> Float {
        _normABuffer.zero()
        gpu.deploy("strictDiff", buffers: [self, vec, _normABuffer], threadCount: self.rows)
        gpu.eval()
        return _normABuffer[0]
    }
    func cosineSimilarityTo(_ vec: Vector) -> ScalarFloat {
        let dotBuffer = ScalarFloat(value:0)
        _normABuffer.zero()
        _normBBuffer.zero()
        gpu.deploy("cosinePrecalc16", buffers: [self, vec, dotBuffer, _normABuffer, _normBBuffer], threadCount: self.rows)
        gpu.deploy("cosineCalc", buffers: [dotBuffer, _normABuffer, _normBBuffer], threadCount: 1)
        gpu.eval()
        return dotBuffer
    }
    
    func scalarAt(_ row: Int) -> Scalar {
        return Scalar(buffer: self.buffer, offset: row)
    }

    func copyFrom32(_ vec: VectorFloat) {
        assert(self.rows == vec.rows)
        gpu.deploy("floatToHalf", buffers: [vec, self], threadCount: self.rows)
    }
    
    func copy() -> Vector {
        let out = Vector(shape:self.shape)
        gpu.deploy("memcpy", buffers: [self, out], threadCount: self.rows)
        return out
    }
    
    func softmax() {
        _rms.zero()
        gpu.deploy("sum_of_exps", buffers: [self, _rms], threadCount: self.rows)
        gpu.deploy("softmax_add", buffers: [self, _rms], threadCount: self.rows)
    }
    
    func rmsNormed() -> Vector {
        let layer = self
        assert(layer.shape.count == 1, "Only for vectors")
        
        let output = Vector(shape: layer.shape)
        gpu.deploy("rms_norm", buffers: [layer, output], ints: [self.count()], threadCount: layer.count())

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
        assert(self.shape == vector.shape, "Shapes of both layers must match")

        gpu.deploy("add_vec", buffers:[self, vector, self], threadCount: self.rows)
    }

    func mul(byVec wa: Vector) {
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


//let sm = ScalarFloat(value: 0)

func calcScores(xq_heads: [Vector], xkTokenHeads: [[Vector]]) -> [Vector] {
    let numTokens = xkTokenHeads.count
    let scores = Matrix(shape: [numHeads, numTokens])

    for t2 in 0..<numTokens {
        for headNo in 0..<numHeads {
            assert(xq_heads[headNo].rows == xkTokenHeads[t2][headNo].rows)
//            gpu.deploy("dotSetScore", buffers: [xq_heads[headNo], xkTokenHeads[t2][headNo], scores.scalarAt(headNo, t2)],
//                       ints: [xq_heads[headNo].rows], threadCount:1)

            assert(xq_heads[headNo].rows == 128, "not tested/implemented for other values.");
            gpu.deploy("dotSetScore2", buffers: [xq_heads[headNo], xkTokenHeads[t2][headNo], scores.scalarAt(headNo, t2)],
                       ints: [1], threadCount:128, threadGroupSize: [128, 1, 1])
 
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

func silu(_ x1: Vector, _ x3: Vector, out: Vector) {
    gpu.deploy("silu", buffers: [x1, x3, out], threadCount: x1.rows)
}

func silu(_ x1: VectorFloat, _ x3: VectorFloat, out: VectorFloat) {
    gpu.deploy("silu32", buffers: [x1, x3, out], threadCount: x1.rows)
}


func bucketMul(v: VectorFloat, by: Weights, out: VectorFloat, quant: Double = 0.25) {
    BucketMul.shared.calcDispatch(v32: v, weights: by, quant: quant)
    BucketMul.shared.mul(by: by, out: out)

}

func bucketMul(v: Vector, by: Weights, out: VectorFloat, quant: Double = 0.25) {
    BucketMul.shared.calcDispatch(v: v, weights: by, quant: quant)
    BucketMul.shared.mul(by: by, out: out)/*
    gpu.eval()
    for i in 0..<out.rows {
        if abs(out[i])>40 {
            print("oh hello", out[i])
            let n = mpsMul(v: v, by: by);
            gpu.eval();
            if out.cosineSimilarityTo(n)[0]<0.90 {
                let disp = BucketMul.shared.dispatch
                print("xx")
            }
            break

        }
    }*/
}

class BucketMul {
    let probesCount = 4096
    let maxDispatchSize = 176128
    let dispatch : DynaVectorFloat
    let probes : Vector
    let cutoff : Scalar
    
    static let shared = BucketMul()
    
    private init() {
        self.dispatch = DynaVectorFloat(shape: [maxDispatchSize*2])
        self.probes = Vector(shape: [probesCount])
        self.cutoff = Scalar(value: 0)
    }
 
    func calcDispatch(v: Vector, weights w: Weights, quant: Double) {
        assert(dispatch.rows >= w.buckets.rows*2)
        dispatch.size.zero()
        
        gpu.deploy("probe", buffers:[v, w.core, probes], ints:[w.inSize], threadCount: probesCount)
        probes.sort()

        let q = Int(Double(probesCount)*(1-quant))
        gpu.deploy("getVal", buffers: [probes, cutoff], ints:[q], threadCount: probesCount)
        //print(cutoff[0])
        let chunkSize = 16//w.stats.rows//16
        gpu.deploy("prepareDispatch", buffers:[v, w.stats, cutoff, dispatch, dispatch.size],
                   ints:[chunkSize, w.inSize], threadCount: w.stats.rows/chunkSize)
    }

    func calcDispatch(v32: VectorFloat, weights w: Weights, quant: Double) {
        let v = v32.asFloat16Vector()
        assert(dispatch.rows >= w.buckets.rows*2)
        dispatch.size.zero()
        
        gpu.deploy("probe", buffers:[v, w.core, probes], ints:[w.inSize], threadCount: probesCount)
        probes.sort()

        let q = Int(Double(probesCount)*(1-quant))
        gpu.deploy("getVal", buffers: [probes, cutoff], ints:[q], threadCount: probesCount)
        let chunkSize = 16//w.stats.rows//16
        gpu.deploy("prepareDispatch32", buffers:[v32, w.stats, cutoff, dispatch, dispatch.size],
                   ints:[chunkSize, w.inSize], threadCount: w.stats.rows/chunkSize)
    }

    
    func mul(by: Weights, out: VectorFloat) {
        let weightBuckets = by.buckets
        
        let bucketSize = 16
        let numBuckets = out.rows / bucketSize
//        assert(weightBuckets.shape == [bucketSize*v.rows, numBuckets], "\(weightBuckets.shape) ≠ \([bucketSize*v.rows, numBuckets])")
        
        assert(numBuckets % 4 == 0)

        let groups = 32
        gpu.deploy("bucketMul", buffers: [weightBuckets, dispatch, out, dispatch.size],
                                ints: [weightBuckets.cols!, groups],
                                threadCount: weightBuckets.cols!,
                                threadCountY:groups)
    }
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
