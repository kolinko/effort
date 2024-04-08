
//
//  q8.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 02/04/2024.
//

import Foundation

struct ConvertOptions: OptionSet {
    let rawValue: Int

    static let mistral = ConvertOptions(rawValue: 1 << 0)
    static let mixtral = ConvertOptions(rawValue: 1 << 1)
    static let q8 = ConvertOptions(rawValue: 1 << 2)
    static let fp16 = ConvertOptions(rawValue: 1 << 3)
}

//func runConvert(mistral: Bool = false, mixtral: Bool = false, Q8: Bool = false, FP16: Bool = true) {
func runConvert(_ options: ConvertOptions) {
        
    if options.contains(.mistral) {
        if options.contains(.fp16) {
            convertMistral(goQ8: false)
        }
        if options.contains(.q8) {
            convertMistral(goQ8: true)
        }
    }
    if options.contains(.mixtral) {
        if options.contains(.fp16) {
            convertMixtral(goQ8: false)
        }
        if options.contains(.q8) {
            convertMixtral(goQ8: true)
        }
    }
    
}


func convertMixtral(goQ8: Bool) {
    let tensors = TensorLoader(path:"./models/mixtral-7x8b")
    let saveTensors = TensorSaver(path:"./models/mixtral-7x8b", model: "buckets-\(goQ8 ? "Q8" : "FP16" )")

    print("done loading.\n")

    let numLayers = 32 // 32
    let numExperts = 8 // 8
    for layerNo in 0..<numLayers {
        var layerTensors = saveTensors[layerNo]
        
        if layerNo == 0 {
            for k in ["model.norm.weight", "lm_head.weight", "model.embed_tokens.weight"] {
                layerTensors[k] = tensors[k]
            }
        }
        
        for k in ["block_sparse_moe.gate.weight",
                  "input_layernorm.weight",
                  "post_attention_layernorm.weight"] {
            let prefix = "model.layers.\(layerNo)."

            layerTensors[prefix+k] = tensors[prefix+k]
        }
        
        
        for s in ["k", "o", "q", "v"] {
            let prefix = "model.layers.\(layerNo).self_attn.\(s)_proj."
//            gpu.startCapture()
            layerTensors[prefix+"weight"] = tensors[prefix+"weight"]
            gpu.stopCapture()

            bucketize(tensors[prefix+"weight"] as! Matrix, outTensorsPref: prefix, tensors: &layerTensors, goQ8: goQ8)
        }
        
        for expertNo in 0..<numExperts {
            var evalTime = Date()
            let prefix = "model.layers.\(layerNo).block_sparse_moe.experts.\(expertNo)."
            print("processing \(prefix)..")
            //let ten = Vector(shape:[4096]); ten.copyFrom((tensors[prefix+"w1.weight"] as! Matrix)[0]); ten.convertBF16();ten.str
            bucketize(tensors[prefix+"w1.weight"] as! Matrix, outTensorsPref: prefix+"w1.", tensors: &layerTensors, goQ8: goQ8)
            gpu.eval()

            bucketize(tensors[prefix+"w2.weight"] as! Matrix, outTensorsPref: prefix+"w2.", tensors: &layerTensors, goQ8: goQ8)
            gpu.eval()

            bucketize(tensors[prefix+"w3.weight"] as! Matrix, outTensorsPref: prefix+"w3.", tensors: &layerTensors, goQ8: goQ8)
            gpu.eval()
            let repsLeft = 8*32-expertNo-(layerNo*8)
            var sumTime = Date().timeIntervalSince(evalTime)
            let totalTime = sumTime * Double(repsLeft)

            // Convert totalTime to minutes and seconds
            let minutes = Int(totalTime) / 60
            let seconds = Int(totalTime) % 60

            print("speed \(Int(sumTime)) seconds.")
            print("ETA: \(minutes) minutes and \(seconds) seconds.")


        }

        saveTensors[layerNo] = layerTensors
    }
  
    saveTensors.save()
}

func bucketize(_ w: Matrix, outTensorsPref: String, tensors: inout [String: MTLBufferable], goQ8: Bool = true) {
    precondition(w.rows >= 4096 || 4096 % w.rows == 0)
    let probeRowRepeat = w.rows >= 4096 ? 1 : 4096 / w.rows
    precondition(w.cols >= 4096)
    // ^ probes not implemented for lesser values
    precondition(w.rows <= 32000, "some dims larger than 32k - \(w.rows), not tested for these sizes")
    precondition(w.cols <= 32000, "some dims larger than 32k - \(w.cols), not tested for these sizes")
    // ^ not tested above max int, may break somewhere
    
    
    let probes = Vector(shape:[4096])
    gpu.deploy("getProbes", buffers: [w, probes], ints: [w.cols, probeRowRepeat], threadCount: 4096/probeRowRepeat)//4096 //1)
    
    // transpose w, convert from bfloats to halfs, create an array of idxs
    let wVals = Matrix(shape:[w.shape[1], w.shape[0]])
    let wIdxs = Matrix(shape:[w.shape[1], w.shape[0]])
    gpu.deploy("prepareValsIdxs", buffers: [w, wVals, wIdxs], ints:[w.rows, w.cols], threadCount: w.rows)
    
    for rowId in 0..<wVals.rows {
        wVals[rowId].sortAbs(idxs: wIdxs[rowId])
    }
    
    let inDim = wVals.rows
    let outDim = wVals.cols
    let bSize = goQ8 ? 8 : 16
    // Q8 works decently on M2 only with buckets sized 8 and possibly below.
    // FP16 works with buckets sized 16 as well.
    // you want the biggest buckets your architecture will allow to get the best
    // perc resolution.
    
    assert(outDim % bSize == 0)
    // group into buckets
//    gpu.startCapture()
    let bVals = Matrix3D(shape:[inDim, outDim/bSize, bSize+1]) // first=counter
    gpu.deploy("preBucketize", buffers:[wVals, wIdxs, bVals], ints:[inDim, outDim, bSize], threadCount: inDim)
    
    // transposition
    let buckets = Matrix(shape:[inDim*bSize, outDim/bSize])
    gpu.deploy("bucketize", buffers: [bVals, buckets], ints:[inDim, outDim, bSize], threadCount: [inDim, outDim/bSize])
    
    // compute averages, to be used by dispatch
    let stats = Vector(shape:[inDim*bSize])
    gpu.deploy("makeStats", buffers: [buckets, stats], ints:[buckets.cols], threadCount: buckets.rows)
    gpu.stopCapture()
    
    tensors[outTensorsPref+"stats"] = stats
    tensors[outTensorsPref+"probes"] = probes
    
    if !goQ8 {
        tensors[outTensorsPref+"buckets"] = buckets
        return
    }
    
    // Q8
    assert(bSize == 8)
    let pBits = 3
    assert(1 << pBits == bSize)
    
    assert (pBits == 3) // hardcoded in many conversion and bucketMulQ8 places
    
    /*
     QUANTISATION
    
     input: fp16, with least significant pBITS bits used to encode the bucket position
     outputted format, for 3 bits of positional encoding
     [SvvvvvPPP]
     where:
     S - sign
     v - integer value
     PPP - position in the bucket
     
     to decode the real value, you need bucket's minRange and diff (stored in the sliceStats)
     FP16 val = sign * (minRange + diff*intVal)
     
     for each row we also keep a set of outliers that are above maxCutoff (first priority)
     and below minCutoff (second priority), and we calc them separately.

     There is probably a better way to implement the outlier selection, but I want to get
     all this shipped finally. I hope someone fixes this.
     
    */
    
    assert(buckets.cols % 2 == 0)
    let bucketsQ8 = Matrix(shape: [buckets.rows, buckets.cols/2])
    let slicesQ8 = bucketsQ8.sliced(numSlices: bSize)
    
    let outliersQ8 = Matrix(shape: [buckets.rows, bSize*2])
    let outlierSlicesQ8 = outliersQ8.sliced(numSlices: bSize)
    
    let slices = buckets.sliced(numSlices: bSize)
    let sliceStatsQ8 = MatrixFloat(shape:[bSize, 2]) // [minRange, diff], for decoding quantisation later on
    
    for i in 0..<bSize {
        let slice = Matrix(shape:slices[i].shape)
        slice.copyFrom(slices[i]) // we'll be messing up original array
        let sliceQ8 = slicesQ8[i]
        
        let minCutoff = findPercentile(v: slice.asVector(), perc: 0.01)
        let maxCutoff = findPercentile(v: slice.asVector(), perc: 0.98)
        
        let minRange = findPercentile(v: slice.asVector(), perc: 0.05)
        let maxRange = findPercentile(v: slice.asVector(), perc: 0.95)
        sliceStatsQ8[i].setVal(Float(minRange), at: 0)
        let numSteps = 16; assert(bSize == 8); // steps = 2^(8-1-log2(bSize))
        sliceStatsQ8[i].setVal((Float(maxRange)-Float(minRange))/Float(numSteps), at: 1)
        
        let outliers = outlierSlicesQ8[i]
/*        print(slice.str)
        print(minRange, maxRange)
        print(findPercentile(v: slice.asVector(), perc: 0.99))
        print(findPercentile(v: slice.asVector(), perc: 0.95))
        print(findPercentile(v: slice.asVector(), perc: 0.90))
        print(findPercentile(v: slice.asVector(), perc: 0.85))
        print(findPercentile(v: slice.asVector(), perc: 0.80))
        print(findPercentile(v: slice.asVector(), perc: 0.75))

        gpu.startCapture()*/
        gpu.deploy("extract", buffers: [slice, sliceQ8, outliers],
                   ints: [slice.cols],
                   floats: [minCutoff, maxCutoff, minRange, maxRange],
                   threadCount: slice.rows)
        gpu.eval()
        gpu.stopCapture()
/*        gpu.stopCapture()
        let o = sliceQ8[0].decodeQ8(minRange: minRange, step: Float16((Float(maxRange)-Float(minRange))/Float(bSize)))
        gpu.eval()
        print(o.str)
        print(slice.str)
        exit(0)*/
    }
    
    tensors[outTensorsPref+"bucketsQ8"] = bucketsQ8
    tensors[outTensorsPref+"outliersQ8"] = outliersQ8
    tensors[outTensorsPref+"sliceStatsQ8"] = sliceStatsQ8

    
}


private let tmpScalarFloat = ScalarFloat(value: 0)

func findPercentile(v: Vector, perc: Double) -> Float {
    // find a given percentile in abs(v)
    // looks at only 4096 els, bc I reused find cutoff that had 4096 hardcoded :p
    assert(v.count>=4096)
    let q = Int(4096 * perc)
    tmpScalarFloat.zero()
    gpu.deploy("findPercentile", buffers: [v, tmpScalarFloat], ints:[q], threadCount: 1024, threadGroupSize: [1024, 1, 1])
    gpu.eval()
    return tmpScalarFloat.val
}


func convertMistral(goQ8: Bool) {
    /*
    let tensors = TensorLoader(path:"./models/mistral")
    let saveTensors = TensorSaver(path:"./models/mistral", model: "buckets-\(goQ8 ? "Q8" : "FP16" )")

    print("done loading.\n")

    let numLayers = 32
    for layerNo in 0..<numLayers {
        var layerTensors = saveTensors[layerNo]
        
        if layerNo == 0 {
            for k in ["model.norm.weight", "lm_head.weight", "model.embed_tokens.weight"] {
                layerTensors[k] = tensors[k]
            }
        }
        
        for k in ["input_layernorm.weight",
                  "post_attention_layernorm.weight"] {
            let prefix = "model.layers.\(layerNo)."

            layerTensors[prefix+k] = tensors[prefix+k]
        }
        
        
        for s in ["k", "o", "q", "v"] {
            let prefix = "model.layers.\(layerNo).self_attn.\(s)_proj."
            bucketize(tensors[prefix+"weight"] as! Matrix, outTensorsPref: prefix, tensors: &layerTensors, goQ8: goQ8)
            layerTensors[prefix+"weight"] = tensors[prefix+"weight"]

        }
        
        let prefix = "model.layers.\(layerNo).mlp."
        print("processing \(prefix)..")
        
        bucketize(tensors[prefix+"up_proj.weight"] as! Matrix, outTensorsPref: prefix+"w1.", tensors: &layerTensors, goQ8: goQ8)
        gpu.eval()

        bucketize(tensors[prefix+"down_proj.weight"] as! Matrix, outTensorsPref: prefix+"w2.", tensors: &layerTensors, goQ8: goQ8)
        gpu.eval()

        bucketize(tensors[prefix+"gate_proj.weight"] as! Matrix, outTensorsPref: prefix+"w3.", tensors: &layerTensors, goQ8: goQ8)
        gpu.eval()
        
        saveTensors[layerNo] = layerTensors
    }
    
    saveTensors.save()
     */
}
