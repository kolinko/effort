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

class MTLBufferable {
    private var _buffer: MTLBuffer? = nil
    private let _fname: String?
    private let _expectedShape: [Int]?
    
    let offsetBytes: Int
    
    init(buffer: MTLBuffer, offsetBytes: Int = 0) {
        self._buffer = buffer
        self.offsetBytes = offsetBytes
        self._fname = nil
        self._expectedShape = nil
    }
    
    init(fname: String, shape: [Int]) {
        self.offsetBytes = 0
        self._fname = fname
        self._expectedShape = shape
    }
    
    var buffer: MTLBuffer {
        if self._buffer != nil {
            return self._buffer!
        } else {
            assert(_fname != nil)
            self._buffer = loadBinaryFile(named: _fname!, shape: _expectedShape!)
            return self._buffer!
        }
    }
    
    func load() {
        _ = self.buffer
    }
    
    func unloadBuffer() {
        self._buffer = nil // does this erase memory?
    }

}

class Bufferable<Type: FloatingPoint> : MTLBufferable {
    var bufferPointer: UnsafeMutablePointer<Type> {
        if self._bufferPointer == nil {
            self._bufferPointer = buffer.contents().bindMemory(to: Type.self, capacity: self.shape.reduce(1, *))
        }
        return self._bufferPointer!
    }
    let shape: [Int]
    let rows: Int
    let cols: Int?
    let byteSize: Int
    let bitSize: Int
    var _bufferPointer : UnsafeMutablePointer<Type>? = nil
    
    var count : Int {
        return self.shape.reduce(1, *)
    }

    override init(fname: String, shape: [Int]) {
        assert(shape.count > 0)
        assert(shape.reduce(1, *) > 0)
        self.byteSize = MemoryLayout<Type>.size
        self.bitSize = byteSize * 8
        assert((byteSize == 4) || (byteSize == 2), "untested for others")
        self.rows = shape[0]
        self.cols = shape.count >= 2 ? shape[1] : nil
        self.shape = shape
        super.init(fname: fname, shape: shape)
    }
    
    init(shape: [Int], buffer: MTLBuffer, offset: Int = 0) {
        assert(shape.count > 0)
        assert(shape.reduce(1, *) > 0)

        self.byteSize = MemoryLayout<Type>.size
        self.bitSize = byteSize * 8

        assert((byteSize == 4) || (byteSize == 2), "untested for others")
        self.rows = shape[0]
        self.cols = shape.count >= 2 ? shape[1] : nil
        self.shape = shape
        super.init(buffer: buffer, offsetBytes: offset*self.byteSize)

    }
    
    convenience init(shape: [Int]) {
        let bufferSize = shape.reduce(1, *) * MemoryLayout<Type>.size
        let buffer = gpu.device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        self.init(shape: shape, buffer: buffer)
    }


    convenience init(shape: [Int], with: Type) {
        self.init(shape: shape)
        for i in 0..<self.count {
            self[i] = with
        }
    }
    
    func zero() {
        gpu.deploy("zero\(bitSize)", buffers: [self], threadCount: self.count)
    }
    func neg() {
        gpu.deploy("neg\(bitSize)", buffers: [self], threadCount: self.count)
    }
    
    func mul(by s: Scalar) {
        gpu.deploy("mulScalar\(bitSize)x16", buffers:[self, s], threadCount:self.count)
    }

    func mul(by s: ScalarFloat) {
        gpu.deploy("mulScalar\(bitSize)x32", buffers:[self, s], threadCount:self.count)
    }

    
    func add(by buf: Bufferable<Type>) {
        assert(self.shape == buf.shape, "Shapes of both buffers must match")

        gpu.deploy("add\(bitSize)", buffers:[self, buf, self], threadCount: self.rows)
    }

    func copyFrom(_ src: Bufferable<Type>) {
        assert(src.count == self.count)
        gpu.deploy("memcpy\(self.bitSize)", buffers: [src, self], threadCount: self.rows)
    }
    
    var str: String {
        return _str()
    }
    
    func _str(count: Int = 10, noEval: Bool = false) -> String {
        if !noEval { gpu.eval() }
        let _count = count<self.rows ? count : self.rows
        var outStr = ""
        for i in 0..<_count {
            outStr += "\(self[i]), "
        }
        return outStr
    }
    
    func getInt(index: Int) -> Int16 {
        var floatStorage: Type
            floatStorage = self[index]

        var intStorage: Int16 = 0

        withUnsafePointer(to: &floatStorage) { floatPointer in
            floatPointer.withMemoryRebound(to: Int16.self, capacity: 1) { intPointer in
                intStorage = intPointer.pointee
            }
        }
        return intStorage
    }

    
    subscript(index: Int) -> Type {
            get {
                let bufferPointer = self.bufferPointer
                return bufferPointer[index+Int(self.offsetBytes/self.byteSize)]
            }
            set(newValue) {
                let bufferPointer = self.bufferPointer
                bufferPointer[index+Int(self.offsetBytes/self.byteSize)] = newValue
            }
        }
    
    
    func test(_ name: String, cond: Bool = true, mul:Int, val:[Type]) -> Bool {
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
        
    func test(mul:Int, val:[Type]) -> Bool {
        gpu.eval()
        for i in 0..<val.count {
            if val[i] != 666 {
                if round(self[i]*Type(mul)) != round(val[i]*Type(mul)) {
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

class ScalarFloat: Bufferable<Float> {
    convenience init(value: Float) {
        self.init(shape: [1])
        self[0] = value;
    }
    
    convenience init(buffer: MTLBuffer, offset: Int = 0) {
        self.init(shape: [1], buffer: buffer, offset: offset)
    }
}


class Scalar: Bufferable<Float16> {
    convenience init(value: Float16) {
        self.init(shape: [1])
        self[0] = value;
    }
    
    convenience init(buffer: MTLBuffer, offset: Int = 0) {
        self.init(shape: [1], buffer: buffer, offset: offset)
    }
}


class Matrix: Bufferable<Float16> {
    func asVector() -> Vector {
        return Vector(shape: [self.count], buffer: self.buffer)
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

class MatrixFloat: Bufferable<Float> {
    func asVector() -> VectorFloat {
        return VectorFloat(shape: [self.count], buffer: self.buffer)
    }
        
    func asVectorList() -> [VectorFloat] {
        var out = [VectorFloat]()
        out.reserveCapacity(self.rows)
        for i in 0..<self.rows {
            out.append(VectorFloat(shape:[self.cols!], buffer:self.buffer, offset: i*self.cols!))
        }
        return out
    }
    
    func scalarAt(_ row: Int, _ col: Int) -> ScalarFloat {
        return ScalarFloat(buffer: self.buffer, offset: row*self.cols! + col)
    }

}

class DynaVectorFloat: VectorFloat {
    let size: ScalarFloat = ScalarFloat(value:0)
}

let _normABuffer = ScalarFloat(value: 0)
let _normBBuffer = ScalarFloat(value: 0)
let _rms = ScalarFloat(value: 0.0)
let _dotBuffer = ScalarFloat(value:0)


class VectorFloat: Bufferable<Float> {
    
    func cosineSimilarityTo(_ vec: VectorFloat) -> Float {
        _normABuffer.zero()
        _normBBuffer.zero()
        _dotBuffer.zero()
        gpu.deploy("cosinePrecalc32", buffers: [self, vec, _dotBuffer, _normABuffer, _normBBuffer], threadCount: self.rows)
        gpu.deploy("cosineCalc32", buffers: [_dotBuffer, _normABuffer, _normBBuffer], threadCount: 1)
        gpu.eval()
        return _dotBuffer[0]
    }

    func strictCompareTo(_ vec: VectorFloat) -> Bool {
        _normABuffer.zero()
        gpu.deploy("strictDiff32", buffers: [self, vec, _normABuffer], threadCount: self.rows)
        gpu.eval()
        return _normABuffer[0] == 0
    }
    
    func asFloat16() -> Vector {
        let out = Vector(shape:[self.rows])
        gpu.deploy("floatToHalf", buffers: [self, out], threadCount: self.rows)
        return out
    }
    
    func copy() -> VectorFloat {
        let out = VectorFloat(shape:self.shape)
        gpu.deploy("memcpy\(self.bitSize)", buffers: [self, out], threadCount: self.rows)
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
    
    func scalarAt(_ row: Int) -> ScalarFloat {
        return ScalarFloat(buffer: self.buffer, offset: row)
    }
    
    func rmsNormed() -> VectorFloat {
        let output = VectorFloat(shape: self.shape)
        gpu.deploy("rmsNorm32", buffers: [self, output], threadCount: self.count)

        return output
    }
    
    func softmax() {
        _rms.zero()
        gpu.deploy("sum_of_exps32", buffers: [self, _rms], threadCount: self.rows)
        gpu.deploy("softmax_add32", buffers: [self, _rms], threadCount: self.rows)
    }
    
    func mul(by wa: Vector) {
        assert(self.shape == wa.shape)
        gpu.deploy("mulVec32by16", buffers:[self, wa, self], threadCount:self.rows)
    }
    
    func mul(complexArray: VectorFloat) {
        assert(self.rows == complexArray.rows, "Layer size must be twice the size of the complex array")
        gpu.deploy("mulComplex32", buffers: [self, complexArray], threadCount: self.rows / 2)
    }
    
    func repeated(_ count: Int) -> VectorFloat {
        assert(self.rows == 128*8)
        let output = VectorFloat(shape: [count*self.rows])
        gpu.deploy("repeat4x32", buffers: [self, output], threadCount: 128, threadCountY: 8)
        return output
    }
    
}


class Vector: Bufferable<Float16> {
    func copy() -> Vector {
        let out = Vector(shape:self.shape)
        gpu.deploy("memcpy\(self.bitSize)", buffers: [self, out], threadCount: self.rows)
        return out
    }

    func asFloat32() -> VectorFloat {
        let out = VectorFloat(shape:[self.rows])
        gpu.deploy("halfToFloat", buffers: [self, out], threadCount: self.rows)
        return out
    }
    
    /*
    func cosineSimilarityTo(_ vec: Vector) -> ScalarFloat {
        let dotBuffer = ScalarFloat(value:0)
        _normABuffer.zero()
        _normBBuffer.zero()
        gpu.deploy("cosinePrecalc16", buffers: [self, vec, dotBuffer, _normABuffer, _normBBuffer], threadCount: self.rows)
        gpu.deploy("cosineCalc", buffers: [dotBuffer, _normABuffer, _normBBuffer], threadCount: 1)
        gpu.eval()
        return dotBuffer
    }*/
    
    func scalarAt(_ row: Int) -> Scalar {
        return Scalar(buffer: self.buffer, offset: row)
    }

    func copyFrom32(_ vec: VectorFloat) {
        assert(self.rows == vec.rows)
        gpu.deploy("floatToHalf", buffers: [vec, self], threadCount: self.rows)
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
    

    func mul(by wa: Vector) {
        assert(self.shape == wa.shape)
        
        gpu.deploy("mulVec16by16", buffers:[self, wa, self], threadCount:self.rows)
    }
    
    func sort() {
        guard let logn = Int(exactly: log2(Double(self.rows))) else {
            fatalError("data.count is not a power of 2")
        }

        var justDispatch = false
        for p in 0..<logn {
            for q in 0..<p+1 {
                gpu.deploy("basicBitonicSort", buffers: [self], ints: [p, q], threadCount: self.rows, justDispatch: justDispatch)
                justDispatch = false
            }
        }
    }
    
}

/*
 
 array funcs
 
 */


func createFreqsCis(headDim: Int, maxSeqLen: Int) -> [VectorFloat] {
    func logspace(start: Double, end: Double, num: Int, base: Double = 10.0) -> [Double] {
        assert(num>1)
        let step = (end - start) / Double(num)
        return (0..<num).map { pow(base, start + Double($0) * step) }
    }

    assert(headDim==128, "unusual headDim. it should work with others, but asserts/tests will fail")
    let freqs = logspace(start: 0, end: 1.0, num: headDim / 2, base: 1e-4)
    assert(freqs[2] == 0.7498942093324559)
    let heads = MatrixFloat(shape: [2*maxSeqLen, freqs.count*2]).asVectorList()
    for i in 0..<(2 * maxSeqLen) {
        for j in 0..<freqs.count {
            let freq = freqs[j]
            let angle = Float(i) * Float(freq)
            let realPart = cos(angle)
            let imagPart = sin(angle)
            heads[i][j*2] = realPart
            heads[i][j*2+1] = imagPart
        }
    }
    assert(heads[1][2]==0.6479058)
    assert(heads[1][3]==0.7617204)
    return heads
}

func calcScores(xq_heads: [VectorFloat], xkTokenHeads: [[VectorFloat]]) -> [VectorFloat] {
    let numTokens = xkTokenHeads.count
    let scores = MatrixFloat(shape: [numHeads, numTokens])

    for t2 in 0..<numTokens {
        for headNo in 0..<numHeads {
            assert(xq_heads[headNo].rows == 128, "not tested/implemented for other values.");
            gpu.deploy("dotSetScore32", buffers: [xq_heads[headNo], xkTokenHeads[t2][headNo], scores.scalarAt(headNo, t2)],
                       ints: [1], threadCount:128, threadGroupSize: [128, 1, 1])
        }
    }
    
    return scores.asVectorList()
}

func gpuConsolidate(vecList src:[VectorFloat]) -> MatrixFloat {
    assert(src.count > 0)

    let out = MatrixFloat(shape:[src.count, src[0].rows])
    let outVecs = out.asVectorList()
    
    for i in 0..<src.count {
        outVecs[i].copyFrom(src[i])
    }

    return out
}

func sumScores(numHeads: Int, headDim:Int, scores: [VectorFloat], xvToken: [VectorFloat]) -> VectorFloat {
    let outMatrix = MatrixFloat(shape: [numHeads, headDim])
    let scoresMatrix = gpuConsolidate(vecList: scores)
    let xvTokenMatrix = gpuConsolidate(vecList: xvToken)

    let numTokens = scores[0].rows
    let numDims = numHeads*headDim
    gpu.deploy("sumScores32", buffers:[scoresMatrix, xvTokenMatrix, outMatrix], ints: [numTokens], threadCount: numDims)
    
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
    out.zero()
    BucketMul.shared.mul(by: by, out: out)

}
/*
func bucketMul(v: Vector, by: Weights, out: VectorFloat, quant: Double = 0.25) {
    BucketMul.shared.calcDispatch(v: v, weights: by, quant: quant)
    out.zero()
    BucketMul.shared.mul(by: by, out: out)
}*/

class BucketMul {
    let probesCount = 4096
    let maxDispatchSize = 229376 * 2//176128
    let dispatch : DynaVectorFloat
    let probes : Vector
    let cutoff : Scalar
    
    static let shared = BucketMul()
    
    private init() {
        self.dispatch = DynaVectorFloat(shape: [maxDispatchSize*2])
        self.probes = Vector(shape: [probesCount])
        self.cutoff = Scalar(value: 0)
    }
 /*
    func calcDispatch(v: Vector, weights w: Weights, quant: Double) {
        assert(dispatch.rows >= w.buckets.rows*2)
        dispatch.size.zero()
        
        assert(w.probes.rows == 4096, "probes implemented for 4096 only. needs review of sort as well as probeShort")
        gpu.deploy("probeShort", buffers:[v, w.probes, probes], ints:[w.inSize], threadCount: probesCount)
        probes.sort()

        let q = Int(Double(probesCount)*(1-quant))
        gpu.deploy("getVal", buffers: [probes, cutoff], ints:[q], threadCount: probesCount)
        gpu.eval()
        print(cutoff[0])
        let chunkSize = 16//w.stats.rows//16
        gpu.deploy("prepareDispatch", buffers:[v, w.stats, cutoff, dispatch, dispatch.size],
                   ints:[chunkSize, w.inSize], threadCount: w.stats.rows/chunkSize)
    }*/

    func calcDispatch(v32: VectorFloat, weights w: Weights, quant: Double) {
        assert(dispatch.rows >= w.buckets.rows*2)
        dispatch.size.zero()
        assert(w.probes.rows == 4096, "probes implemented for 4096 only. needs review of sort as well as probeShort")
        gpu.deploy("probeShort", buffers:[v32, w.probes, probes], ints:[w.inSize], threadCount: probesCount)
        probes.sort()

        let q = Int(Double(probesCount-1)*(1-quant))
        gpu.deploy("getVal", buffers: [probes, cutoff], ints:[q], threadCount: probesCount)
        let chunkSize = 16//w.stats.rows//16
        gpu.deploy("prepareDispatch32", buffers:[v32, w.stats, cutoff, dispatch, dispatch.size],
                   ints:[chunkSize, w.inSize], threadCount: w.stats.rows/chunkSize)
    }

    
    func mul(by: Weights, out: VectorFloat) {
        let weightBuckets = by.buckets
        
        let bucketSize = 16
        let numBuckets = out.rows / bucketSize
        
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
