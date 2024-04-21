/*
 
 Needs a bit of refactoring probably.
 
 bucketMulFast is the main wrapper function, kept similar in calling style to basicMul, so it can be
 quickly replaced in other places in code.

 */


func bucketMulFast(v: VectorFloat, by: ExpertWeights, expNo: ScalarFloat, out: VectorFloat, effort: Double = 0.25) {
    let bm = BucketMulFast.shared
    bm.fullMul(v: v, ew: by, expNo: expNo, out: out, effort: effort)
}

// should be private, but this way is useful for testing
class BucketMulFast {
    let probesCount = 4096
    let maxDispatchSize = 229376 * 2//176128
    let dispatch : DynaVectorFloat
    let probes : Vector
    let cutoff : ScalarFloat
    private let prevSize = ScalarFloat(value: 0)
    
    static let shared = BucketMulFast()
    
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
      //  print("dsize", dispatch.size.getLong(index: 0))
    }
    
    
    private let mulGroups = 32
    private let tmpMulVec = MatrixFloat(shape:[32, 16384])

    func fullMul(v: VectorFloat, ew: ExpertWeights, expNo: ScalarFloat, out: VectorFloat, effort: Double) {
        calcDispatch(v: v, eWeights: ew, expNo: expNo, effort: effort)
        
        gpu.deploy("roundUp", buffers:[dispatch.size, prevSize], ints:[2048], threadCount: 1)
        gpu.deploy("zeroRange32", buffers: [dispatch, prevSize, dispatch.size], threadCount: 2048 )
        // ^ quick patch here.
        // bucketMulFast goes through dispatch in bucketSize * STEP chunks, and if dispatchSize is not evened
        // out, the ranges may start to overlap and cause subtle errors at various Effort levels.
        
        // not sure if 2048 rounding is right at this iteration, needs testing and fixing probably
        
        mul(by: ew, out: out)
    }

    
   func mul(by: ExpertWeights, out: VectorFloat) {
       let weightBuckets = by.buckets
       
       assert(!goQ8, "call BucketMulQ8, not this")
       let bucketSize = 16
       let numBuckets = out.rows / bucketSize
       
       assert(numBuckets % 4 == 0)

       gpu.deploy("bucketMulFast", buffers: [weightBuckets, dispatch, tmpMulVec, dispatch.size],
                               ints: [weightBuckets.cols, mulGroups],
                               threadCount: [weightBuckets.cols, mulGroups])
       
       let simdSize = 32
       gpu.deploy("bucketIntegrate", buffers: [tmpMulVec, out],
                  threadCount: [simdSize, out.rows/4, 1],
                  threadGroupSize: [simdSize, 1, 1])
        

   }
    
}

