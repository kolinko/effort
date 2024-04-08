//
//  q8.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 02/04/2024.
//

import Foundation



func expertMulQ8v3(v: VectorFloat, by: ExpertWeights, expNo: ScalarFloat, out: VectorFloat, quant: Double = 0.25) {
//    return
    let bm = BucketMulQ8.shared
    out.zero()
    bm.calcDispatch(v: v, eWeights: by, expNo: expNo, quant: quant)
//    gpu.eval()
    bm.mulv4(ew: by, out: out)
}

/*
func expertMulQ8v3(v: VectorFloat, by: ExpertWeightsQ8, expNo: ScalarFloat, out: VectorFloat, quant: Double = 0.25) {
    out.zero()
    BucketMul.shared.calcDispatch(v: v, eWeights: by, expNo: expNo, quant: quant)
    gpu.deploy("round", buffers:[BucketMul.shared.dispatch.size], ints:[1024], threadCount: 1) // tofix
    BucketMul.shared.mul3(by: by, out: out)
    /*
    timeIt(repeats:10000) { i in     //  max possible = 10000. Good enough = 5000.
        gpu.deploy("setVal", buffers: [expNo], ints:[i % 8], threadCount: 1)
        BucketMul.shared.calcDispatch(v: v, eWeights: by, expNo: expNo, quant: quant)
        BucketMul.shared.mul3(by: by, out: out)
    }

    exit(0)*/
}
*/



class BucketMulQ8 {
    let probesCount = 4096
    let maxDispatchSize = 229376 * 2//176128
    let dispatch : DynaVectorFloat
    let probes : Vector
    let cutoff : Scalar
    
    static let shared = BucketMulQ8()
    
    private init() {
        self.dispatch = DynaVectorFloat(shape: [maxDispatchSize*4])
        self.probes = Vector(shape: [probesCount])
        self.cutoff = Scalar(value: 0)
    }
        
    func calcDispatch(v: VectorFloat, eWeights ew: ExpertWeights, expNo: ScalarFloat, quant: Double) {
        assert(dispatch.rows >= ew.buckets.rows*2)
        assert(ew.probes.cols == 4096, "probes implemented for 4096 only. needs review of sort as well as probeShort")

        let sliceStats = ew.sliceStats!
        
        dispatch.size.zero()
        let q = Int(Double(probesCount-1)*(1-quant))

        gpu.deploy("findCutoff", buffers: [v, ew.probes, expNo, cutoff], ints:[q], threadCount: 1024, threadGroupSize: [1024, 1, 1])

        
        let chunkSize = 1
        gpu.deploy("prepareExpertDispatchQ8", buffers:[v, ew.stats, expNo, cutoff, dispatch, dispatch.size, sliceStats],
                   ints:[chunkSize, ew.inSize, ew.buckets.cols, ew.expertSize], threadCount: ew.buckets.rows/chunkSize)
    }
    
    


    
    private let mulGroups = 64
    private let tmpMulVec = MatrixFloat(shape:[64, 16384])
    
    func mulv4(ew: ExpertWeights, out: VectorFloat) {
        assert(out.rows < 16384) // tmpMulVec max size hardcoded

        let weightBucketsQ8 = ew.buckets
        
        let bucketSize = 8
        let numBuckets = out.rows / bucketSize
        assert(numBuckets % 4 == 0)

        gpu.deploy("bucketMulQ8v4", buffers: [weightBucketsQ8, dispatch, tmpMulVec, dispatch.size],
                   ints: [weightBucketsQ8.cols, mulGroups],
                   threadCount: [weightBucketsQ8.cols, mulGroups, 1],
                   threadGroupSize: [64,1,1])
        
        gpu.deploy("bucketIntegrate", buffers: [tmpMulVec, out],
                   threadCount: [32, out.rows, 1],
                   threadGroupSize: [32, 1, 1])
        
    }

    func mulv3(ew: ExpertWeights, out: VectorFloat) {

        let weightBucketsQ8 = ew.buckets
        
        let bucketSize = 8
        let numBuckets = out.rows / bucketSize
        assert(numBuckets % 4 == 0)

        let groups = 64

        gpu.deploy("bucketMulQ8v3", buffers: [weightBucketsQ8, dispatch, out, dispatch.size],
                   ints: [weightBucketsQ8.cols, groups],
                   threadCount: [weightBucketsQ8.cols, groups, 1],
                   threadGroupSize: [64,1,1])
    }

    
    func mul(ew: ExpertWeights, out: VectorFloat) {
        
        let weightBucketsQ8 = ew.buckets
        
        let bucketSize = 8
        let numBuckets = out.rows / bucketSize
        assert(numBuckets % 4 == 0)

        let groups = 64
        gpu.deploy("bucketMulQ8", buffers: [weightBucketsQ8, dispatch, out, dispatch.size],
                   ints: [weightBucketsQ8.cols, groups],
                   threadCount: [weightBucketsQ8.cols, groups, 1],
                   threadGroupSize: [64,1,1])
        
//        gpu.deploy("bucketMulQ8integrate", buffers:[tmpOut, out], )
    }

    
    /*
     
     let groups = 64

     gpu.deploy("bucketMul3", buffers: [weightBuckets, dispatch, out, dispatch.size],
                             ints: [weightBuckets.cols, groups*2],
                             threadCount: [weightBuckets.cols, groups, 1],
                             threadGroupSize: [64,1,2])
     
     --> 0.09ms for Q8!
     */
    
    
    
    
}

func expertMulQ8(v: VectorFloat, by: ExpertWeights, expNo: ScalarFloat, out: VectorFloat, quant: Double = 0.25) {
    let bm = BucketMulQ8.shared
    out.zero()
    bm.calcDispatch(v: v, eWeights: by, expNo: expNo, quant: quant)
    bm.mul(ew: by, out: out)
    return
}


/*
 
     func mulOutliers(ew: ExpertWeightsQ8, out: VectorFloat) {
         let outliers = ew.outliersQ8
         let groups = 128
         let maxOutliers = 32
         gpu.deploy("bucketOutliersQ8",
                    buffers: [outliers, dispatch, out, dispatch.size],
                    ints: [groups],
                    threadCount: [maxOutliers, groups],
                    threadGroupSize: [32, 1, 1])
     }

 */
