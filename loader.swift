/*
 
 This went through so many iterations, and needs cleanup. Should be renamed Model really.

 Weight loading is handled by helpers/loader.
 
 Core = basic weight matrix, used by MPSMul. Everything else is related to buckets.
 
*/

import Metal
import Foundation
let shapeDict = readJson()

/*
 
 BAM = queries each buffer every 100ms to prevent system from unloading it to swap.
 Without it, if a model reaches 80% occupied memory, it will be offloaded/loaded into
 memory all the time, swap will kick in and you'll get one token per minute performance.
 You can see the struggle in the system Activity Monitor, Memory usage, and looking at Wired/Compressed mem stats.
 
 MLX is missing this btw, and that's why it's generation speeds are crap for larger models.
 
 There is a system setting to increase allowed wired memory size, but BAM seems to work better.
 
 */

private let bam = BufferActivityManager()

let modelPath = "./models/\(goMistral ? "mistral" : "mixtral-new")"
let modelName = "buckets-\(goQ8 ? "Q8" : "FP16")"
let jsonPath = "./"

private let tLoader = TensorLoader(path: modelPath, model: modelName)

class Weights {
    let core: Matrix
    var outSize: Int { return core.rows }
    var inSize: Int { return core.cols }
    
    init(elName: String, shape: [Int]? = nil) {
        self.core = tLoader.matrix(elName+".core", assertShape: (shape != nil) ? shape : shapeDict[elName]!)
    }
}

/*
 
 ExpertWeights need to handle the following
 - storing both core weights as an option to bucket weights (sometimes you want both easily accessible for testing)
 - regular, and expert weights - you want to split them here, because all expert weights need to be passed as one buffer to BucketMul
 - Q8 and FP16 - Q8 has an extra matrix of SliceStats
 
 Originally I implemented it as separate classes, but then the rest of the code was more messed up.
 */

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

/*
 
 I dislike this approach to loading - passing file/weight names to down-layers,
 but it ended up as the most elegant for the time being.
 
 Neater solutions welcome!
 
 */

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

/*
 
 Core loading below.
 
 Q: Can't we use multithreading to load faster?
 
    I tried, ended up with the same speed, but uglier.
    The hard drive speed is a limiting factor here if I measured correctly. Cannot be optimised for speed.
 
 A: Can't we use memmap, and the system will load everything when it needs it?
  
    It uses memmap underneath, but forces to load everything here and now, because otherwise it looks ugly when
    getting stuck in random places, BAM cannot be implemented easily and so on.
 
 I also experimented with dynamic loading of buffers when needed (to be able to load both core and bucket weights
 and just dynamically choose which ones are needed), but the memory management became a mess.
 
 */

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

func testSetup(_ modelsDir: String, _ modelDir: String, _ modelIndex: String) {
    
    ensureDirectoryExists(for: "./", createDirectoryAtPath: modelsDir)
    ensureDirectoryExists(for: "./\(modelsDir)/", createDirectoryAtPath: modelDir)

    let modelIndex = "./\(modelsDir)/\(modelDir)/\(modelIndex)"
    if !FileManager.default.fileExists(atPath: modelIndex) {
        print("\nModel data not found at \(modelIndex)")
        print("\nIf running from XCode and you have the model already:\n>>>   edit scheme -> working directory -> project directory\n")
        print("If running from terminal:")
        print(">>>  huggingface-cli download kolinko/mistral-buckets --exclude \"*Q8*\" --local-dir ./\(modelsDir)/\(modelDir)")
        print("")
        print("If you don't have huggingface CLI:")
        print(">>>  pip install -U \"huggingface_hub[cli]\"")
        print()
        exit(0)
    }
    
}

func ensureDirectoryExists(for path: String, createDirectoryAtPath createPath: String) {
    let fileManager = FileManager.default
    let exists = fileManager.fileExists(atPath: path + createPath)

    if !exists {
        do {
            try fileManager.createDirectory(atPath: path+createPath, withIntermediateDirectories: true)
        } catch {
            print("Failed to create directory: \(error)")
        }
    }
}



func physicalMemoryGB() -> Int {
    return Int(ProcessInfo.processInfo.physicalMemory / 1024 / 1024 / 1024)

}

func autoAdjustPercent(max: Int) -> Int {
    if max != 0x10 {
        return max
    }
    
    var percentLoad = max
    print("Physical Memory: \(physicalMemoryGB()) GB")

    if physicalMemoryGB() <= 8 {
        print("\nWhat is this? A computer for ants?\n\nI'll load just 37% of weights, the answers will be barely understandable.")
        print("Q8 is in the works and it will require just half the mem and give ~twice the speed, hopefully.\n")
        print("Press Enter to continue")
        _ = readLine()
        percentLoad = 0xB
    } else if physicalMemoryGB() <= 16 {
        print("\nAw! You're a bit short on memory.\nI'll load just 75% of the model, ok? Quality will suffer, but it should run without swap then.")
        print("Q8 is in the works and it will require just half the mem and give ~twice the speed, hopefully.\n")
        print("Press Enter to continue")
        _ = readLine()
        percentLoad = 0xB
    }

    return percentLoad
}
