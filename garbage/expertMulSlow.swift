//
//  expertMulSlow.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 17/04/2024.
//

import Foundation

/*
 
 Original, unused implementation, but well tested. Still sometimes useful, so stays
 as a comment, but will be removed.
 
 */

/*
 
func expertMulSlow(v: VectorFloat, by: ExpertWeights, expNo: ScalarFloat, out: VectorFloat, effort: Double = 0.25) {
    if goNoMuls {return;}
    out.zero()
    BucketMul.shared.calcDispatch(v: v, eWeights: by, expNo: expNo, effort: effort)
//    print(BucketMul.shared.dispatch.size.val, BucketMul.shared.dispatch.size.intVal)
    BucketMul.shared.mul(by: by, out: out)
    return
}

func expertMul3(v: VectorFloat, by: ExpertWeights, expNo: ScalarFloat, out: VectorFloat, effort: Double = 0.25) {
    out.zero()
    BucketMul.shared.calcDispatch(v: v, eWeights: by, expNo: expNo, effort: effort)
    gpu.deploy("round", buffers:[BucketMul.shared.dispatch.size], ints:[1024], threadCount: 1) // tofix
    BucketMul.shared.mul3(by: by, out: out)
    /*
    timeIt(repeats:10000) { i in     //  max possible = 10000. Good enough = 5000.
        gpu.deploy("setVal", buffers: [expNo], ints:[i % 8], threadCount: 1)
        BucketMul.shared.calcDispatch(v: v, eWeights: by, expNo: expNo, effort: effort)
        BucketMul.shared.mul3(by: by, out: out)
    }

    exit(0)*/
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
        
    func calcDispatch(v: VectorFloat, eWeights ew: ExpertWeights, expNo: ScalarFloat, effort: Double) {
        assert(dispatch.rows >= ew.buckets.rows*2)
        assert(ew.probes.cols == 4096, "probes implemented for 4096 only. needs review of sort as well as probeShort")

        dispatch.size.zero()
        let q = Int(Double(probesCount-1)*(1-effort))

        gpu.deploy("findCutoff", buffers: [v, ew.probes, expNo, cutoff], ints:[q], threadCount: 1024, threadGroupSize: [1024, 1, 1])
        
        let chunkSize = 16//w.stats.rows//16
        gpu.deploy("prepareExpertDispatch", buffers:[v, ew.stats, expNo, cutoff, dispatch, dispatch.size],
                   ints:[chunkSize, ew.inSize, ew.buckets.cols, ew.expertSize], threadCount: ew.stats.rows/chunkSize)
    }
    
    func mul3(by: ExpertWeights, out: VectorFloat) {
        let weightBuckets = by.buckets
        
        let bucketSize = 16
        let numBuckets = out.rows / bucketSize
        
        assert(numBuckets % 4 == 0)

        let groups = 128

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

*/

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
