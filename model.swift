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

class Bufferable<Type> : MTLBufferable {
    var bufferPointer: UnsafeMutablePointer<Type> {
        if self._bufferPointer == nil {
            self._bufferPointer = buffer.contents().bindMemory(to: Type.self, capacity: self.shape.reduce(1, *))
        }
        return self._bufferPointer!
    }
    var shape: [Int] // FIX BACK TO LET!
    var rows: Int {self.shape[0]}
    let byteSize: Int
    let bitSize: Int
    var countBytes: Int {self.count * self.byteSize}
    var offsetEls: Int {assert(self.offsetBytes % self.byteSize == 0); return self.offsetBytes/self.byteSize}

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
        self.shape = shape
        super.init(fname: fname, shape: shape)
    }
    
    init(shape: [Int], buffer: MTLBuffer, offset: Int = 0) {
        assert(shape.count > 0)
        assert(shape.reduce(1, *) > 0)

        self.byteSize = MemoryLayout<Type>.size
        self.bitSize = byteSize * 8

        assert((byteSize == 4) || (byteSize == 2), "untested for others")
        self.shape = shape
        super.init(buffer: buffer, offsetBytes: offset*self.byteSize)

    }
    
    convenience init(shape: [Int], private _private: Bool = false) {
        let bufferSize = shape.reduce(1, *) * MemoryLayout<Type>.size
        var _buffer: MTLBuffer
        if _private {
            _buffer = gpu.device.makeBuffer(length: bufferSize, options: .storageModePrivate)!
        } else {
            _buffer = gpu.device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        }
//        self.buffer = _buffer
        self.init(shape: shape, buffer: _buffer)
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

        gpu.deploy("add\(bitSize)", buffers:[self, buf, self], threadCount: self.count)
    }

    func copyFrom(_ src: Bufferable<Type>) {
        assert(src.count == self.count)
        gpu.copyBuffer(src: src, dst: self, size: src.countBytes)
    }
    
    var str: String {
        return _str()
    }
    
    
    var strInt: String {
        gpu.eval()
        let _count = count<self.count ? count : self.count
        var outStr = ""
        for i in 0..<_count {
            outStr += "\(self.getInt(index: i)), "
        }
        return outStr
    }
    
    
    func _str(count: Int = 10, noEval: Bool = false) -> String {
        if !noEval { gpu.eval() }
        let _count = count<self.count ? count : self.count
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
    
    /*
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
    }*/
}

class ScalarFloat: Bufferable<Float> {
    convenience init(value: Float) {
        self.init(shape: [1])
        self[0] = value;
    }
    
    convenience init(buffer: MTLBuffer, offset: Int = 0) {
        self.init(shape: [1], buffer: buffer, offset: offset)
    }
    
    var val : Float {self[0]}
    var intVal: Int16 {self.getInt(index: 0)}
}


class Scalar: Bufferable<Float16> {
    convenience init(value: Float16) {
        self.init(shape: [1])
        self[0] = value;
    }
    
    convenience init(buffer: MTLBuffer, offset: Int = 0) {
        self.init(shape: [1], buffer: buffer, offset: offset)
    }

    var val : Float16 {self[0]}
    var intVal: Int16 {self.getInt(index: 0)}

}


class Matrix: Bufferable<Float16> {
    var cols: Int { self.shape[1] }
    
    func asVector() -> Vector {
        return Vector(shape: [self.count], buffer: self.buffer, offset: self.offsetEls)
    }
    
    func scalarAt(_ row: Int, _ col: Int) -> Scalar {
        return Scalar(buffer: self.buffer, offset: self.offsetEls + row*self.cols + col)
    }
    
    func asVectorList() -> [Vector] {
        var out = [Vector]()
        out.reserveCapacity(self.rows)
        for i in 0..<self.rows {
            out.append(Vector(shape:[self.cols], buffer:self.buffer, offset: self.offsetEls+i*self.cols))
        }
        return out
    }
    
    subscript(index: Int) -> Vector {
            get {
                Vector(shape:[self.cols], buffer:self.buffer, offset: self.offsetEls+index*self.cols)
            }
        }

}

class Matrix3DFloat: Bufferable<Float> {
    override var rows: Int { self.shape[1] }
    var cols: Int { self.shape[2] }
    var slices: Int { self.shape[0] }
//    var sliceSize: Int {self.rows * self.count}

    func asMatrixList() -> [MatrixFloat] {
        assert(self.shape.count == 3)
        var out = [MatrixFloat]()
        out.reserveCapacity(self.slices)
        for i in 0..<self.slices {
            out.append(MatrixFloat(shape:[shape[1], shape[2]], buffer:self.buffer, offset: self.offsetEls + i*self.shape[1]*self.shape[2]))
        }
        return out
    }
    
    subscript(index: Int) -> MatrixFloat {
            get {
                return MatrixFloat(shape:[shape[1], shape[2]], buffer:self.buffer, offset: self.offsetEls + index*self.shape[1]*self.shape[2])
            }
        }
}

class Matrix4DFloat: Bufferable<Float> {
    var cols: Int { self.shape[3] }
    override var rows: Int { self.shape[2] }
    var slices: Int { self.shape[1] }
    var sliceGroups: Int {self.shape[0]}

    func as3DMatrixList() -> [Matrix3DFloat] {
        assert(self.shape.count == 4)
        var out = [Matrix3DFloat]()
        out.reserveCapacity(self.sliceGroups)
        for i in 0..<self.sliceGroups {
            out.append(Matrix3DFloat(shape:[shape[1], shape[2], shape[3]], buffer:self.buffer, offset: self.offsetEls + i*self.shape[1]*self.shape[2]*self.shape[3]))
        }
        return out
    }
    
    subscript(index: Int) -> Matrix3DFloat {
            get {
                return Matrix3DFloat(shape:[shape[1], shape[2], shape[3]], buffer:self.buffer, offset: self.offsetEls + index*self.shape[1]*self.shape[2]*self.shape[3])
            }
        }
}

class Matrix3D: Bufferable<Float16> {
    override var rows: Int { self.shape[1] }
    var cols: Int { self.shape[2] }
    var slices: Int { self.shape[0] }
//    var sliceSize: Int {self.rows * self.count}

    func asMatrixList() -> [Matrix] {
        assert(self.shape.count == 3)
        var out = [Matrix]()
        out.reserveCapacity(self.slices)
        for i in 0..<self.slices {
            out.append(Matrix(shape:[shape[1], shape[2]], buffer:self.buffer, offset: self.offsetEls + i*self.shape[1]*self.shape[2]))
        }
        return out
    }
}

class MatrixFloat: Bufferable<Float> {
    var cols: Int { self.shape[1] }

    func asVector() -> VectorFloat {
        return VectorFloat(shape: [self.count], buffer: self.buffer, offset: self.offsetEls)
    }
        
    func asVectorList() -> [VectorFloat] {
        var out = [VectorFloat]()
        out.reserveCapacity(self.rows)
        for i in 0..<self.rows {
            out.append(VectorFloat(shape:[self.cols], buffer:self.buffer, offset: self.offsetEls + i*self.cols))
        }
        return out
    }
    
    subscript(index: Int) -> VectorFloat {
            get {
                VectorFloat(shape:[self.cols], buffer:self.buffer, offset: self.offsetEls + index*self.cols)
            }
        }
    
    func scalarAt(_ row: Int, _ col: Int) -> ScalarFloat {
        return ScalarFloat(buffer: self.buffer, offset: self.offsetEls + row*self.cols + col)
    }

}

class DynaVectorFloat: VectorFloat {
    let size: ScalarFloat = ScalarFloat(value:0)
    
    func bins(binSize: Int) -> [Int] {
        gpu.eval()
        var bins = [Int](repeating: 0, count: 16)
        for i in 0..<Int(self.size.val) {
            bins[(Int(self[i*2+1]) % binSize * 16)/binSize] += 1
        }
        for i in 0..<16 {
            bins[i] = (bins[i]*100)/Int(self.size.val)
        }

        return bins
    }
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
    
    func asMatrix(newCols: Int) -> MatrixFloat {
        assert(self.rows % newCols == 0, "Original layer size must be divisible by new dimension size")
        let newRows = self.rows / newCols

        return MatrixFloat(shape:[newRows, newCols], buffer: self.buffer, offset: self.offsetEls)
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
        gpu.deploy("repeat4x32", buffers: [self, output], threadCount: [128, 8])
        return output
    }

    func repeated(_ count: Int, into:VectorFloat) {
        assert(self.rows == 128*8)
        assert(into.rows == count*self.rows)
//        let output = VectorFloat(shape: [count*self.rows])
        gpu.deploy("repeat4x32", buffers: [self, into], threadCount: [128, 8])
//        return output
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


func calcScores2(xq_heads: MatrixFloat, xkTokenHeads: Matrix3DFloat, numTokens: Int) -> MatrixFloat {
    let scores = MatrixFloat(shape: [numHeads, numTokens])

    gpu.deploy("dotSetScore2",
               buffers: [xq_heads, xkTokenHeads, scores],
               threadCount: [128, numTokens, numHeads],
               threadGroupSize: [128, 1, 1])
    
    return scores
}

func calcScores(xq_heads: [VectorFloat], xkTokenHeads: [[VectorFloat]]) -> [VectorFloat] {
    let numTokens = xkTokenHeads.count
    let scores = MatrixFloat(shape: [numHeads, numTokens])

    for t2 in 0..<numTokens {
        for headNo in 0..<numHeads {
            assert(xq_heads[headNo].rows == 128, "not tested/implemented for other values.");
            gpu.deploy("dotSetScore32", buffers: [xq_heads[headNo], xkTokenHeads[t2][headNo], scores.scalarAt(headNo, t2)],
                       ints: [1], threadCount:[128], threadGroupSize: [128, 1, 1])
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

func sumScores2(numHeads: Int, headDim:Int, scores: MatrixFloat, xvToken: MatrixFloat, numTokens: Int) -> VectorFloat {
    let outMatrix = MatrixFloat(shape: [numHeads, headDim])
    let numDims = numHeads*headDim
    
    assert(scores.cols == numTokens)
    
    gpu.deploy("sumScores32", buffers:[scores, xvToken, outMatrix], ints: [numTokens], threadCount: [numDims])
    
    return outMatrix.asVector()
}

func sumScores(numHeads: Int, headDim:Int, scores: [VectorFloat], xvToken: [VectorFloat]) -> VectorFloat {
    let outMatrix = MatrixFloat(shape: [numHeads, headDim])
    let scoresMatrix = gpuConsolidate(vecList: scores)
    let xvTokenMatrix = gpuConsolidate(vecList: xvToken)

    let numTokens = scores[0].rows
    let numDims = numHeads*headDim
    gpu.deploy("sumScores32", buffers:[scoresMatrix, xvTokenMatrix, outMatrix], ints: [numTokens], threadCount: [numDims])
    
    return outMatrix.asVector()
}

func silu(_ x1: Vector, _ x3: Vector, out: Vector) {
    gpu.deploy("silu", buffers: [x1, x3, out], threadCount: x1.rows)
}

func silu(_ x1: VectorFloat, _ x3: VectorFloat, out: VectorFloat) {
    gpu.deploy("silu32", buffers: [x1, x3, out], threadCount: x1.rows)
}


func expertMul(v: VectorFloat, by: ExpertWeights, expNo: ScalarFloat, out: VectorFloat, quant: Double = 0.25) {
    out.zero()
    BucketMul.shared.calcDispatch2(v: v, eWeights: by, expNo: expNo, quant: quant)
    BucketMul.shared.mul2(by: by, out: out)

//    return
    //    expNo[0] = 3;
    //  max possible = 10000. Good enough = 5000.
    var goTime = Date()
    if true {
        gpu.eval()
    //    gpu.startCapture()
        gpu.eval()}
    for i in 0..<10 {
        gpu.deploy("setVal", buffers: [expNo], ints:[i % 8], threadCount: 1)
        BucketMul.shared.calcDispatch3(v: v, eWeights: by, expNo: expNo, quant: quant)
        BucketMul.shared.mul3(by: by, out: out)
    }
    gpu.eval()
   // gpu.stopCapture()
    let numLoops = 10000
    for i in 0..<numLoops {
        gpu.deploy("setVal", buffers: [expNo], ints:[i % 8], threadCount: 1)
        BucketMul.shared.calcDispatch3(v: v, eWeights: by, expNo: expNo, quant: quant)
        BucketMul.shared.mul3(by: by, out: out)
    }
    print("prep time \(Date().timeIntervalSince(goTime)*1000, precision: 2) ms")
    goTime = Date()
    gpu.eval()
    print("final eval time \(Date().timeIntervalSince(goTime)*1000, precision: 2) ms")
    print("eval per loop \(Date().timeIntervalSince(goTime)*1000/Double(numLoops), precision: 2) ms")

    print("persec \(Double(numLoops) / Date().timeIntervalSince(goTime), precision: 2) runs")
    exit(0)
}

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

    func calcDispatch3(v: VectorFloat, eWeights ew: ExpertWeights, expNo: ScalarFloat, quant: Double) {
        assert(dispatch.rows >= ew.buckets.rows*2)
        assert(ew.probes.cols == 4096, "probes implemented for 4096 only. needs review of sort as well as probeShort")

        dispatch.size.zero()
        let q = Int(Double(probesCount-1)*(1-quant))

        gpu.deploy("findCutoff", buffers: [v, ew.probes, expNo, cutoff], ints:[q], threadCount: 1024, threadGroupSize: [1024, 1, 1])

        
        let chunkSize = 16//w.stats.rows//16
        gpu.deploy("prepareExpertDispatch", buffers:[v, ew.stats, expNo, cutoff, dispatch, dispatch.size],
                   ints:[chunkSize, ew.inSize, ew.expertSize], threadCount: ew.stats.rows/chunkSize)
        gpu.deploy("round", buffers:[dispatch.size], ints:[1024], threadCount: 1) // tofix
    }
    
    func calcDispatch2(v: VectorFloat, eWeights ew: ExpertWeights, expNo: ScalarFloat, quant: Double) {
        assert(dispatch.rows >= ew.buckets.rows*2)
        assert(ew.probes.cols == 4096, "probes implemented for 4096 only. needs review of sort as well as probeShort")

        dispatch.size.zero()
        let q = Int(Double(probesCount-1)*(1-quant))

        gpu.deploy("findCutoff", buffers: [v, ew.probes, expNo, cutoff], ints:[q], threadCount: 1024, threadGroupSize: [1024, 1, 1])

        
        let chunkSize = 16//w.stats.rows//16
        gpu.deploy("prepareExpertDispatch", buffers:[v, ew.stats, expNo, cutoff, dispatch, dispatch.size],
                   ints:[chunkSize, ew.inSize, ew.expertSize], threadCount: ew.stats.rows/chunkSize)
        gpu.deploy("round", buffers:[dispatch.size], ints:[1024], threadCount: 1) // tofix
    }
    
    func calcDispatch(v: VectorFloat, eWeights ew: ExpertWeights, expNo: ScalarFloat, quant: Double) {
        assert(dispatch.rows >= ew.buckets.rows*2)
        assert(ew.probes.cols == 4096, "probes implemented for 4096 only. needs review of sort as well as probeShort")

        dispatch.size.zero()
        let q = Int(Double(probesCount-1)*(1-quant))

        if runControl == true {
            gpu.deploy("probeExpert", buffers:[v, ew.probes, expNo, probes], ints:[ew.inSize], threadCount: probesCount)
            probes.sort()
            
            gpu.deploy("getVal", buffers: [probes, cutoff], ints:[q], threadCount: probesCount)
        } else {
            gpu.deploy("findCutoff", buffers: [v, ew.probes, expNo, cutoff], ints:[q], threadCount: 1024, threadGroupSize: [1024, 1, 1])
        }
        /*
        gpu.eval()
        if abs(cutoff.val - cutoff2.val)>0.001 {
            print("?=", cutoff.str, cutoff2.str)
        }*/
        
        let chunkSize = 16//w.stats.rows//16
        gpu.deploy("prepareExpertDispatch", buffers:[v, ew.stats, expNo, cutoff, dispatch, dispatch.size],
                   ints:[chunkSize, ew.inSize, ew.expertSize], threadCount: ew.stats.rows/chunkSize)
    }
    
    func mul3(by: ExpertWeights, out: VectorFloat) {
        let weightBuckets = by.buckets
        
        let bucketSize = 16
        let numBuckets = out.rows / bucketSize
        
        assert(numBuckets % 4 == 0)

        let groups = 256

        gpu.deploy("bucketMul3", buffers: [weightBuckets, dispatch, out, dispatch.size],
                                ints: [weightBuckets.cols, groups],
                                threadCount: [weightBuckets.cols, groups, 1],
                                threadGroupSize: [128,1,1])
    }

    /*
     
     let groups = 64

     gpu.deploy("bucketMul3", buffers: [weightBuckets, dispatch, out, dispatch.size],
                             ints: [weightBuckets.cols, groups*2],
                             threadCount: [weightBuckets.cols, groups, 1],
                             threadGroupSize: [64,1,2])
     
     --> 0.09ms for Q8!
     */
    
    
    
    func mul2(by: ExpertWeights, out: VectorFloat) {
        let weightBuckets = by.buckets
        
        let bucketSize = 16
        let numBuckets = out.rows / bucketSize
        
        assert(numBuckets % 4 == 0)

        let groups = 32
        gpu.deploy("bucketMul2", buffers: [weightBuckets, dispatch, out, dispatch.size],
                                ints: [weightBuckets.cols, groups],
                                threadCount: [weightBuckets.cols,groups])
    }

    func mul(by: ExpertWeights, out: VectorFloat) {
        let weightBuckets = by.buckets
        
        let bucketSize = 16
        let numBuckets = out.rows / bucketSize
        
        assert(numBuckets % 4 == 0)

        let groups = 32
        gpu.deploy("bucketMul", buffers: [weightBuckets, dispatch, out, dispatch.size],
                                ints: [weightBuckets.cols, groups],
                                threadCount: [weightBuckets.cols,groups])
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
