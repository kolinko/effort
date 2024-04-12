
private let prevSize = ScalarFloat(value: 0)

func bucketMulFast(v: VectorFloat, by: ExpertWeights, expNo: ScalarFloat, out: VectorFloat, quant: Double = 0.25) {
    if goNoMuls {return;}
  //  out.zero()
    let bm = BucketMulFast.shared
   // bm.dispatch.zero()
    bm.calcDispatch(v: v, eWeights: by, expNo: expNo, quant: quant)
    gpu.eval()   
    print(bm.dispatch.size.long)
//    print(bm.cutoff.val)
    gpu.deploy("roundUp", buffers:[bm.dispatch.size, prevSize], ints:[2048], threadCount: 1) // tofix
    gpu.deploy("zeroRange32", buffers: [bm.dispatch, prevSize, bm.dispatch.size], threadCount: 2048 )
    bm.mul(by: by, out: out)
//    if out.hasNan {
//        print("hello")
//    }
}


class BucketMulFast {
    let probesCount = 4096
    let maxDispatchSize = 229376 * 2//176128
    let dispatch : DynaVectorFloat
    let probes : Vector
    let cutoff : ScalarFloat
    
    static let shared = BucketMulFast()
    
    private init() {
        self.dispatch = DynaVectorFloat(shape: [maxDispatchSize*2])
        self.probes = Vector(shape: [probesCount])
        self.cutoff = ScalarFloat(value: 0)
    }
        
    func calcDispatch(v: VectorFloat, eWeights ew: ExpertWeights, expNo: ScalarFloat, quant: Double) {
        assert(dispatch.rows >= ew.buckets.rows*2)
        assert(ew.probes.cols == 4096, "probes implemented for 4096 only. needs review of sort as well as probeShort")

        dispatch.size.zero()
        
        let q = Int(Double(probesCount-1)*(1-quant))

        gpu.startCapture()
        gpu.deploy("findCutoff32", buffers: [v, ew.probes, expNo, cutoff], ints:[q], threadCount: 1024, threadGroupSize: [1024, 1, 1])
        gpu.stopCapture()
        
        let chunkSize = 4//w.stats.rows//16
        gpu.deploy("prepareExpertDispatchFast", buffers:[v, ew.stats, expNo, cutoff, dispatch, dispatch.size],
                   ints:[chunkSize, ew.inSize, ew.buckets.cols, ew.expertSize], threadCount: ew.stats.rows/chunkSize)
    }
    
    
    private let mulGroups = 32
    private let tmpMulVec = MatrixFloat(shape:[32, 16384])
    
   func mul(by: ExpertWeights, out: VectorFloat) {
       let weightBuckets = by.buckets
       
       let bucketSize = 16
       let numBuckets = out.rows / bucketSize
       
       assert(numBuckets % 4 == 0)

       //let groups = 32
//       dispatch.zero()
//       tmpMulVec.zero()
       gpu.deploy("bucketMulFast", buffers: [weightBuckets, dispatch, tmpMulVec, dispatch.size],
                               ints: [weightBuckets.cols, mulGroups],
                               threadCount: [weightBuckets.cols, mulGroups])
       
       
       let simdSize = 32
       gpu.deploy("bucketIntegrate", buffers: [tmpMulVec, out],
                  threadCount: [simdSize, out.rows/4, 1],
                  threadGroupSize: [simdSize, 1, 1])
        

   }

        /*
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
    }*/

    /*
     
     let groups = 64

     gpu.deploy("bucketMul3", buffers: [weightBuckets, dispatch, out, dispatch.size],
                             ints: [weightBuckets.cols, groups*2],
                             threadCount: [weightBuckets.cols, groups, 1],
                             threadGroupSize: [64,1,2])
     
     --> 0.09ms for Q8!
     */
 
    
}

