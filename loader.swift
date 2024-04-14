/*
 
    Loading weights code
 
 */

import Metal
import Foundation
let shapeDict = readJson()

private let bam = BufferActivityManager()

let modelPath = "./models/\(goMistral ? "mistral" : "mixtral-new")"
let modelName = "buckets-\(goQ8 ? "Q8" : "FP16")"
let jsonPath = "/Users/kolinko/mul_col/model-mixtral/"

private let tLoader = TensorLoader(path: modelPath, model: modelName)


class Weights {
    let core: Matrix
    var outSize: Int { return core.rows }
    var inSize: Int { return core.cols }
    
    init(elName: String, shape: [Int]? = nil) {
        self.core = tLoader.matrix(elName+".core", assertShape: (shape != nil) ? shape : shapeDict[elName]!)
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
    let core: Matrix?
    
    init(elName: String) {
        self.core = tLoader.matrix(elName+".core")
        self.inSize = core!.cols
        self.outSize = core!.rows
        self.percentLoad = goQ8 ? 8 : 16
        
        let probesCount = 4096
        self.probes = Matrix(shape: [1, probesCount])
        let probesList: [Vector] = probes.asVectorList()
        
        self.buckets = Matrix3D(shape: [1, inSize*percentLoad, outSize/16])
        let bucketList: [Matrix] = self.buckets.asMatrixList()
        bam.addBuffer(self.buckets)
        
        self.stats = Matrix3D(shape: [1, inSize*percentLoad, 4])
        let statList: [Matrix] = self.stats.asMatrixList()
        bam.addBuffer(self.stats)

        var sliceStatsList: [MatrixFloat]?
        
        if goQ8 {
            self.sliceStats = Matrix3DFloat(shape: [1, 8 , 2])
            sliceStatsList = self.sliceStats!.asMatrixList()
        } else {
            self.sliceStats = nil
        }
        
        probesList[0].copyFrom(tLoader.vector(elName+".probes"))//, mySize: true)
        bucketList[0].copyFrom(tLoader.matrix(elName+".buckets"))//, mySize: true)
        statList[0].copyFrom(tLoader.matrix(elName+".bucket.stats"))//, mySize: true)
        if goQ8 {
            sliceStatsList![0].copyFrom(tLoader.matrix(elName+".sliceStats"))//, mySize: true)
        }

    }

    
    init(_ prefix: String, _ wId: String? = nil, inDim: Int, outDim: Int, numExperts: Int, percentLoad: Int) {
        self.core = nil
        
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
            self.sliceStats = Matrix3DFloat(shape: [numExperts, 8 , 2])
            sliceStatsList = self.sliceStats!.asMatrixList()
        } else {
            self.sliceStats = nil
        }
        
        for eNo in 0..<numExperts {
            let fName = prefix + (wId != nil ? "\(eNo).\(wId!)." : "")
            probesList[eNo].copyFrom(tLoader[fName+"probes"], mySize: true)
            bucketList[eNo].copyFrom(tLoader[fName+"buckets"], mySize: true)
            statList[eNo].copyFrom(tLoader[fName+"bucket.stats"], mySize: true)
            if goQ8 {
                sliceStatsList![eNo].copyFrom(tLoader[fName+"sliceStats"], mySize: true)
            }
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

    let wo: ExpertWeights
    let wq: ExpertWeights
    let wk: ExpertWeights
    let wv: ExpertWeights

    let w1: ExpertWeights
    let w2: ExpertWeights
    let w3: ExpertWeights
    
    subscript(index: String) -> Matrix {
        get { data[index]! }
        set { data[index] = newValue }
    }
        
    init(_ layerNo: Int, numExperts: Int, percentLoad: Int) {
        let layerName = "layers.\(layerNo)."
        
        self.ffnNorm = tLoader.vector(layerName+"ffn_norm", assertShape:shapeDict[layerName+"ffn_norm"]!)
        self.attnNorm = tLoader.vector(layerName+"attention_norm", assertShape: shapeDict[layerName+"attention_norm"]!)

        if numExperts > 1 {
            self.ffnGate = tLoader.matrix(layerName+"feed_forward.gate", assertShape: [numExperts, stateDim])
        } else {
            self.ffnGate = nil
        }
        
        self.wo = ExpertWeights(elName: layerName+"attention.wo")
        self.wk = ExpertWeights(elName: layerName+"attention.wk")
        self.wq = ExpertWeights(elName: layerName+"attention.wq")
        self.wv = ExpertWeights(elName: layerName+"attention.wv")

        self.numExperts = numExperts
        self.percentLoad = percentLoad
        
        let prefix = "layers.\(layerNo).feed_forward.experts."
        self.w1 = ExpertWeights(prefix, "w1", inDim: stateDim, outDim: hiddenDim, numExperts: numExperts, percentLoad: percentLoad)
        self.w3 = ExpertWeights(prefix, "w3", inDim: stateDim, outDim: hiddenDim, numExperts: numExperts, percentLoad: percentLoad)
        self.w2 = ExpertWeights(prefix, "w2", inDim: hiddenDim, outDim: stateDim, numExperts: numExperts, percentLoad: percentLoad)
    }

}


class Model {
    let norm: Vector
    let output: Weights
    let tokEmbeddings: Matrix
    let layers: [Int: Layer]
    
    init(numLayers: Int, numExperts: Int, percentLoad: Int) {
        let startTime = Date()
        self.norm = tLoader.vector("model.norm", assertShape: shapeDict["norm"]!)
        self.output = Weights(elName: "output", shape: shapeDict["output"]!)
        self.tokEmbeddings = tLoader.matrix("tok_embeddings.core", assertShape: shapeDict["tok_embeddings"]!)

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
        gpu.eval()
    }
}


func readJson() -> [String: [Int]] {
    let fileUrl = URL(fileURLWithPath: jsonPath + "index.json")
    let data = try! Data(contentsOf: fileUrl)
    let dictionary = try! JSONSerialization.jsonObject(with: data, options: []) as! [String: [Int]]

    return dictionary
}

