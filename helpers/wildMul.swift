//
//  wildMul.swift
//  effort
//

/*
 
 This is an experimental, "wild" version, meant to try out doing things by the book - loading weights
 into threadgroup memory first and only then processing the data.
 
 It works like crap.
 
 */

import Foundation


private let prevSize = ScalarFloat(value: 0)

func bucketMulWild(v: VectorFloat, by: ExpertWeights, expNo: ScalarFloat, out: VectorFloat, effort: Double = 0.25) {
    let bm = BucketMulWild.shared
    bm.fullMul(v: v, ew: by, expNo: expNo, out: out, effort: effort)
}

class BucketMulWild {
    let probesCount = 4096
    let maxDispatchSize = 229376 * 2//176128
    let dispatch : DynaVectorFloat
    let probes : Vector
    let cutoff : ScalarFloat
    
    static let shared = BucketMulWild()
    
    private init() {
        self.dispatch = DynaVectorFloat(shape: [maxDispatchSize*2])
        self.probes = Vector(shape: [probesCount])
        self.cutoff = ScalarFloat(value: 0)
    }
        
    func calcDispatch(v: VectorFloat, eWeights ew: ExpertWeights, expNo: ScalarFloat, effort: Double) {
        assert(dispatch.rows >= ew.buckets.rows*2)
        assert(ew.probes.cols == 4096, "probes implemented for 4096 only. needs review of sort as well as probeShort")

        dispatch.size.zero()
        let q = Int(Double(probesCount-1)*(1-effort))

        gpu.deploy("findCutoff32", buffers: [v, ew.probes, expNo, cutoff], ints:[q], threadCount: 1024, threadGroupSize: [1024, 1, 1])
        
        let chunkSize = 4//w.stats.rows//16
        gpu.deploy("prepareExpertDispatchFast", buffers:[v, ew.stats, expNo, cutoff, dispatch, dispatch.size],
                   ints:[chunkSize, ew.inSize, ew.buckets.cols, ew.expertSize], threadCount: ew.stats.rows/chunkSize)
      //  gpu.eval()
        
    //    print("dsize", dispatch.size.getLong(index: 0))
    }
    
    
    private let mulGroups = 32
    private let tmpMulVec = MatrixFloat(shape:[32, 16384])

    func fullMul(v: VectorFloat, ew: ExpertWeights, expNo: ScalarFloat, out: VectorFloat, effort: Double) {
        calcDispatch(v: v, eWeights: ew, expNo: expNo, effort: effort)
        
        gpu.deploy("roundUp", buffers:[dispatch.size, prevSize], ints:[2048], threadCount: 1)
        gpu.deploy("zeroRange32", buffers: [dispatch, prevSize, dispatch.size], threadCount: 2048 )
        
        mul(by: ew, out: out)
    }

    
   func mul(by: ExpertWeights, out: VectorFloat) {
       let weightBuckets = by.buckets
       
       let bucketSize = 16
       let numBuckets = out.rows / bucketSize
       
       assert(numBuckets % 4 == 0)
       let mulGroups = 32;

       gpu.deploy("bucketMulWild", buffers: [weightBuckets, dispatch, out, dispatch.size],
                               ints: [weightBuckets.cols, mulGroups],
                               threadCount: [weightBuckets.cols*16, mulGroups],
                               threadGroupSize: [32*32, 1, 1])
   }

}

