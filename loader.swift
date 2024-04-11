/*
 
    Loading weights code
 
 */

import Metal
import Foundation
let shapeDict = readJson()

private let bam = BufferActivityManager()

class Weights {
    let core: Matrix
    var outSize: Int { return core.rows }
    var inSize: Int { return core.cols }
    
    init(core: Matrix, buckets:Matrix, stats:Matrix, probes: Vector) {
        self.core = core
    }
    
    init(elName: String) {
        let shape = shapeDict[elName]!
        self.core = Matrix(fname: elName+".core.bin", shape: shape)

    }
    
    
    init(fromFile: String, shape: [Int]) {
        self.core = loadBinaryMatrix(named: fromFile+".core.bin", shape: shape)
    }
    
}


class ExpertWeights {
    let inSize: Int
    let outSize: Int
    let percentLoad: Int
    var expertSize: Int {percentLoad*inSize}
    
    let buckets: Matrix3D
    let stats: Matrix3D
    let probes: Matrix
    let sliceStats: Matrix3DFloat?
    
    init(_ wId: String, inDim: Int, outDim: Int, layerNo: Int, numExperts: Int, percentLoad: Int) {//}, Q8: Bool = false) {
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

        var sliceStatsList: [MatrixFloat]?
        
        if goQ8 {
            self.sliceStats = Matrix3DFloat(shape: [numExperts, 8 , 2])//inDim*percentLoad, 1])
            sliceStatsList = self.sliceStats!.asMatrixList()
        } else {
            self.sliceStats = nil
        }
        
        for eNo in 0..<numExperts {
            let fName = "layers.\(layerNo).feed_forward.experts.\(eNo).\(wId)."
            probesList[eNo].copyFrom(tLoader[fName+"probes.bin"], mySize: true)
            bucketList[eNo].copyFrom(tLoader[fName+"buckets.bin"], mySize: true)
            statList[eNo].copyFrom(tLoader[fName+"bucket.stats.bin"], mySize: true)
            if goQ8 {
                sliceStatsList![eNo].copyFrom(tLoader[fName+"sliceStats.bin"], mySize: true)
//                    loadBinaryMatrixFloat(named: fName+"sliceStats.bin", shape: sliceStatsList![eNo].shape))
            }
            /*
             //                loadBinaryVector(named: fName+"probes.bin", shape: [probesCount]))
                         bucketList[eNo].copyFrom(loadBinaryMatrix(named: fName+"buckets.bin", shape: bucketList[eNo].shape))
                         statList[eNo].copyFrom(loadBinaryMatrix(named: fName+"bucket.stats.bin", shape: statList[eNo].shape))
                         if goQ8 {
                             sliceStatsList![eNo].copyFrom(loadBinaryMatrixFloat(named: fName+"sliceStats.bin", shape: sliceStatsList![eNo].shape))
                         }

             */
        }
    }
}

class Layer {
    var data = [String: Matrix]()
    let numExperts: Int
    let percentLoad: Int

    let attnNorm: Vector
    let ffnNorm: Vector
    let ffnGate: Matrix?

    
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
        if numExperts > 1 {
            self.ffnGate = Matrix(fname: layerName+"feed_forward.gate.bin", shape: [numExperts, stateDim])//shapeDict[layerName+"feed_forward.gate"]!)
        } else {
            self.ffnGate = nil
        }
        
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
    init(numLayers: Int, numExperts: Int, percentLoad: Int) {
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
let modelPath = "./models/\(goMistral ? "mistral" : "mixtral-new")"
let modelName = "buckets-\(goQ8 ? "Q8" : "FP16")"
private let tLoader = TensorLoader(path: modelPath, model: modelName)

func loadBuffer(named fileName: String) -> MTLBuffer {
   return tLoader[fileName].buffer
}

func loadBinaryMatrix(named fileName: String, shape: [Int]) -> Matrix {
    return Matrix(shape: shape, buffer: tLoader[fileName].buffer)
}

func loadBinaryMatrixFloat(named fileName: String, shape: [Int]) -> MatrixFloat {
    return MatrixFloat(shape: shape, buffer: tLoader[fileName].buffer)
}

func loadBinaryVector(named fileName: String, shape: [Int]) -> Vector {
    return Vector(shape: shape, buffer: tLoader[fileName].buffer)
}
