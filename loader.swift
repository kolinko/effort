/*
 
    Loading weights code
 
 */

import Metal
import Foundation
let shapeDict = readJson()

class Weights {
    let core: Matrix
    let buckets: Matrix
    let stats: Matrix
    let probes: Vector
    var outSize: Int { return core.rows }
    var inSize: Int { return core.cols }
    
    init(core: Matrix, buckets:Matrix, stats:Matrix, probes: Vector) {
        self.core = core
        self.buckets = buckets
        self.stats = stats
        self.probes = probes
       // assertDims()
    }
    
    init(elName: String) {
        let shape = shapeDict[elName]!
        self.core = Matrix(fname: elName+".core.bin", shape: shape)
        self.buckets = Matrix(fname: elName+".buckets.bin", shape: [shape[1]*16, shape[0]/16])
        self.stats = Matrix(fname: elName+".bucket.stats.bin", shape: [shape[1]*16, 4])
        var probesSize = 4096
        if self.core.rows < 4096 || self.core.cols < 4096 {
            probesSize = 1024
        }
        self.probes = Vector(fname: elName+".probes.bin", shape: [probesSize])
       // assertDims()
    }
    
    func assertDims() {
        assert(core.cols*16 == buckets.rows)
        assert(core.cols*16 == stats.rows)
        if self.inSize >= 4096 && self.outSize >= 4096 {
            assert(self.probes.rows == 4096)
        } else {
            assert(self.probes.rows == 1024)
        }
    }
    
    convenience init(fromFile: String, shape: [Int]) {
        let core : Matrix = loadBinaryMatrix(named: fromFile+".core.bin", shape: shape)
        let bShape = [shape[1]*16, shape[0]/16]
        let buckets : Matrix = Matrix(shape:[1])//loadBinaryMatrix(named: fromFile+".buckets.bin", shape: bShape)
        
        let sShape = [shape[1]*16, 4]
        let stats : Matrix = Matrix(shape:[1])//loadBinaryMatrix(named: fromFile+".bucket.stats.bin", shape: sShape)
        
        let pShape = [(shape[0]>=4096 && shape[1]>=4096) ? 4096 : 1024]
        let probes : Vector = Vector(shape:[1])//loadBinaryMatrix(named: fromFile+".probes.bin", shape: pShape).asVector()

        self.init(core: core, buckets: buckets, stats: stats, probes: probes)
        
    }
    
}

let stateDim = 4096
let hiddenDim = 14336

class ExpertWeights {
    let inSize: Int
    let outSize: Int
    let percentLoad: Int
    var expertSize: Int {percentLoad*inSize}
    
    let buckets: Matrix3D
    let stats: Matrix3D
    let probes: Matrix
    
    init(_ wId: String, inDim: Int, outDim: Int, layerNo: Int, numExperts: Int, percentLoad: Int) {
        self.inSize = inDim
        self.outSize = outDim
        self.percentLoad = percentLoad
        
        let probesCount = 4096

        self.probes = Matrix(shape: [numExperts, probesCount])
        let probesList: [Vector] = probes.asVectorList()
        
        self.buckets = Matrix3D(shape: [numExperts, inDim*percentLoad, outDim/16])
        let bucketList: [Matrix] = self.buckets.asMatrixList()
        bam.addBuffer(self.buckets)
        
        self.stats = Matrix3D(shape: [numExperts, inDim*percentLoad, 4])
        let statList: [Matrix] = self.stats.asMatrixList()
        bam.addBuffer(self.stats)

        for eNo in 0..<numExperts {
            let fName = "layers.\(layerNo).feed_forward.experts.\(eNo).\(wId)."
            probesList[eNo].copyFrom(loadBinaryVector(named: fName+"probes.bin", shape: [probesCount]))
            bucketList[eNo].copyFrom(loadBinaryMatrix(named: fName+"buckets.bin", shape: bucketList[eNo].shape))
            statList[eNo].copyFrom(loadBinaryMatrix(named: fName+"bucket.stats.bin", shape: statList[eNo].shape))
        }
    }
}




class Layer {
    var data = [String: Matrix]()
    let numExperts: Int
    let percentLoad: Int

    let attnNorm: Vector
    let ffnNorm: Vector
    let ffnGate: Matrix

    
    let wo: Weights
    let wq: Weights
    let wk: Weights
    let wv: Weights

    let w1: ExpertWeights
    let w2: ExpertWeights
    let w3: ExpertWeights

    
    subscript(index: String) -> Matrix {
        get { data[index]! }
        set { data[index] = newValue }
    }
        
    init(_ layerNo: Int, numExperts: Int, percentLoad: Int) {
        let layerName = "layers.\(layerNo)."
        
        self.ffnNorm = Vector(fname: layerName+"ffn_norm.bin", shape: shapeDict[layerName+"ffn_norm"]!)
        self.attnNorm = Vector(fname: layerName+"attention_norm.bin", shape: shapeDict[layerName+"attention_norm"]!)
        self.ffnGate = Matrix(fname: layerName+"feed_forward.gate.bin", shape: [numExperts, stateDim])//shapeDict[layerName+"feed_forward.gate"]!)
        
        self.wo = Weights(elName: layerName+"attention.wo")
        self.wk = Weights(elName: layerName+"attention.wk")
        self.wq = Weights(elName: layerName+"attention.wq")
        self.wv = Weights(elName: layerName+"attention.wv")

        self.numExperts = numExperts
        self.percentLoad = percentLoad
        
        self.w1 = ExpertWeights("w1", inDim: stateDim, outDim: hiddenDim, layerNo: layerNo, numExperts: numExperts, percentLoad: percentLoad)
        self.w3 = ExpertWeights("w3", inDim: stateDim, outDim: hiddenDim, layerNo: layerNo, numExperts: numExperts, percentLoad: percentLoad)

        self.w2 = ExpertWeights("w2", inDim: hiddenDim, outDim: stateDim, layerNo: layerNo, numExperts: numExperts, percentLoad: percentLoad)
    }

}

let absolutePath = "/Users/kolinko/mul_col/model-mixtral/"
class Model {
    let norm: Matrix
    let output: Weights
    let tokEmbeddings: Matrix
    let layers: [Int: Layer]
    
    //numLayers: 32, numExperts: 8, percentLoad: 0xA
    init(from filePath: String, numLayers: Int, numExperts: Int, percentLoad: Int) {
        let startTime = Date()

        self.norm = loadBinaryMatrix(named: "norm.bin", shape: shapeDict["norm"]!)
        self.output = Weights(fromFile: "output", shape: shapeDict["output"]!)
        self.tokEmbeddings = loadBinaryMatrix(named: "tok_embeddings.core.bin", shape: shapeDict["tok_embeddings"]!)

        print("loading weights")
        var layers = [Int: Layer]()
        for i in 0..<numLayers {
            layers[i] = Layer(i, numExperts:numExperts, percentLoad: percentLoad)
            print("preparing layer \(i)...\r", terminator:"")
            fflush(stdout)
            gpu.eval()
        }
        self.layers = layers

        print("data init time \(Date().timeIntervalSince(startTime)) seconds")
    }
    
}


func readJson() -> [String: [Int]] {
    let fileUrl = URL(fileURLWithPath: absolutePath + "index.json")
    let data = try! Data(contentsOf: fileUrl)
    let dictionary = try! JSONSerialization.jsonObject(with: data, options: []) as! [String: [Int]]

    return dictionary
}

//private let tLoader = TensorLoader(path: "./model-mixtral", model: "rawfp16")
private let tLoader = TensorLoader(path:"./models/mixtral-new", model: "buckets-FP16")

func loadBinaryFile(named fileName: String, shape: [Int]) -> MTLBuffer {
   // print(fileName)
   // print(shape)
    /*
    var cShape = shape
    
    if shape[0] == 12288 {
        cShape[0] = 65536
    } else if shape[0] == 43008 {
        cShape[0] = 229376
    }
    
    prepConvertBinaryFile(named: fileName, shape: cShape, tSaver: tSaver)
    let targetSuffix = "buckets.bin"
    let replacementSuffix = "core.bin"

    if fileName.hasSuffix(targetSuffix) && fileName !=  "output.buckets.bin" {
        let newFname = fileName.dropLast(targetSuffix.count) + replacementSuffix
        prepConvertBinaryFile(named: String(newFname), shape: [16*shape[0]/3, shape[1]*16], tSaver: tSaver)
      //  print("conv buckets, new size", [shape[1]*16, shape[0]/3])
        if !fileName.contains("experts") {
        //    print("hm")
        }
//        assert(shape.reduce(1, *) ==[shape[1]*16, shape[0]/3] )
    }
     */
    
    let safeten = tLoader[fileName]
    print(fileName, (safeten as! Bufferable<Float16>).shape)
    //assert(shape == [2, 4096] || shape == (safeten as! Bufferable<Float16>).shape)
    return safeten.buffer
    
     
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
    
    var o : MTLBufferable = Vector(shape:[1])
    
    if shape.count == 1 {
        o = Vector(shape: shape, buffer: buffer)
    } else if shape.count == 2 {
        o = Matrix(shape: shape, buffer: buffer)
    } else {
        assert(false)
    }
    
   // print(fileName, shape)
    var layNo : Int? = extractNumber(from: fileName)
    if layNo == nil {
       layNo = 0
    }
    
    assert((o as! Bufferable<Float16>).shape != [1])
//    tSaver[layNo!][fileName] = o
   // print(layNo!)
    close(fileDescriptor)
    return buffer
}



func loadBinaryMatrix(named fileName: String, shape: [Int]) -> Matrix {
    return Matrix(shape: shape, buffer: loadBinaryFile(named: fileName, shape: shape))
}

func loadBinaryVector(named fileName: String, shape: [Int]) -> Vector {
    do {
        return Vector(shape: shape, buffer: loadBinaryFile(named: fileName, shape: shape))
    } catch {
        return Vector(shape: shape, buffer: loadBinaryFile(named:  String(fileName.dropLast(4)), shape: shape))

    }
}

func extractNumber(from string: String) -> Int? {
    do {
        let regex = try NSRegularExpression(pattern: "\\d+", options: [])
        let matches = regex.matches(in: string, options: [], range: NSRange(location: 0, length: min(12, string.count)))
        
        if let match = matches.first {
            if let range = Range(match.range, in: string) {
                let numberString = string[range]
                if let number = Int(numberString) {
                    // Ensure the extracted number is within the range of 0 to 32
                    if (0...32).contains(number) {
                        return number
                    }
                }
            }
        }
    } catch {
        print("Error: \(error)")
    }
    
    return nil
}

