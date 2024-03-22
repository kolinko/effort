//
//  matrix.metal
//  mul_col
//
//  Created by Tomasz Kolinko on 26/01/2024.
//

#include <metal_stdlib>
using namespace metal;

kernel void silu(device const float *x1 [[buffer(0)]],
                 device const float *x3 [[buffer(1)]],
                 device half *out [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
    out[id] = x3[id] * x1[id] / (1 + exp(-x1[id]));
}

/*for i in 0..<probes {
    o[i] = abs(v[i] * weights[i*weights.cols! + i])
}*/

kernel void probe(device const half *v [[buffer(0)]],
                  device const half *weights [[buffer(1)]],
                  device half *out[[buffer(2)]],
                  constant int &wCols [[buffer(3)]],
                  uint id [[thread_position_in_grid]]) {
    out[id] = abs(v[id] * weights[id*wCols + id]);
}

kernel void getVal(device const half* vector [[buffer(0)]],
                   device half *val [[buffer(1)]],
                   const device uint &pos [[buffer(2)]],
                   uint id [[thread_position_in_grid]]) {
    if (id == pos) {
        val[0] = vector[pos];
    }
}

#define SIZE 4096/4
// #define testChunksY 4

kernel void testBucket(device const half2x4 *weights [[buffer(0)]],
                       device half *result [[buffer(1)]],
                       constant uint &rows [[buffer(2)]],
                       constant uint &cols [[buffer(3)]],
                       constant uint &testChunksY [[buffer(4)]],
                       ushort2 id [[thread_position_in_grid]],
                       ushort2 tpig [[threadgroup_position_in_grid]],
                       ushort tpisg [[thread_index_in_simdgroup]]
                       ) {
    half out = 0;
  
    
    const ushort wX = tpisg / 8;
    const ushort wY = (tpisg % 8) / 4;
    const ushort wZ = (tpisg % 8) % 4;
    const ushort offset = tpig.x;// + wX;
    const ushort offRows = tpig.y*11008/16;//rows/testChunksY;
/*
 

 */
//    for (uint r=offRows; r<offRows+rows/testChunksY; r+=4) {
//        ushort r = offRows + i;
    for (uint i = 0; i<11008/16; i+=4) {
        ushort r = offRows + i;
        out += weights[r*cols + offset][wY][wZ];
        out += weights[(r+1)*cols + offset][wY][wZ];
        out += weights[(r+2)*cols + offset][wY][wZ];
        out += weights[(r+3)*cols + offset][wY][wZ];
    }
    result[id.x] = out;
//    result[id / 8][id % 8 / 4][id % 4] = id;//out;

    //    result[tpig*32+tpisg][0][1] = 33;
}

/*
 FAST!
 
 kernel void testBucket(device const half2x4 *weights [[buffer(0)]],
                        device half *result [[buffer(1)]],
                        constant uint &rows [[buffer(2)]],
                        constant uint &cols [[buffer(3)]],
                        constant uint &testChunksY [[buffer(4)]],
                        ushort2 id [[thread_position_in_grid]],
                        ushort2 tpig [[threadgroup_position_in_grid]],
                        ushort tpisg [[thread_index_in_simdgroup]]
                        ) {
     half out = 0;
   
     
     const ushort wX = tpisg / 8;
     const ushort wY = (tpisg % 8) / 4;
     const ushort wZ = (tpisg % 8) % 4;
     const ushort offset = tpig.x*2;// + wX;
     const ushort offRows = tpig.y*rows/testChunksY;
     for (uint r=offRows; r<offRows+rows/testChunksY; r+=8) {
         out += weights[r*cols + offset][wY][wZ];
         out += weights[(r+1)*cols + offset][wY][wZ];
         out += weights[(r+2)*cols + offset][wY][wZ];
         out += weights[(r+3)*cols + offset][wY][wZ];

         
     }
     result[id.x] = out;
 //    result[id / 8][id % 8 / 4][id % 4] = id;//out;

     //    result[tpig*32+tpisg][0][1] = 33;
 }

 */

/*
kernel void testBucket(device const half2x4 *weights [[buffer(0)]],
                       device half2x4 *result [[buffer(1)]],
                       constant ushort &rows [[buffer(2)]],
                       constant ushort &cols [[buffer(3)]],

                       ushort tpig [[threadgroup_position_in_grid]],
                       ushort tpisg [[thread_index_in_simdgroup]]
                       ) {
    float out = 0;
    const ushort offset = tpig*32;// + tpisg);

    for (int r=0; r<rows; r++) {
        out += weights[r*cols + offset][0][0];
    }
    result[tpig*32+tpisg][0][0] = out;
    result[tpig*32+tpisg][0][1] = 33;
} */

kernel void truthBucket(device const half *weights [[buffer(0)]],
                       device half *result [[buffer(1)]],
                       constant ushort &rows [[buffer(2)]],
                       constant ushort &cols [[buffer(3)]],

                       ushort tpig [[thread_position_in_grid]]
                       ) {
    float out = 0;
    const ushort offset = tpig;// + tpisg);
    for (int r=0; r<rows; r++) {
        out += weights[r*cols + offset];
    }
    result[tpig] = out;
}


kernel void truthBucket2(device const half *weights [[buffer(0)]],
                       device half *result [[buffer(1)]],
                       constant ushort &rows [[buffer(2)]],
                       constant ushort &cols [[buffer(3)]],

                       ushort2 tpig [[thread_position_in_grid]]
                       ) {
    half out[16];
    for (int i = 0; i<16; i++) { out[i] = 0;};
    const ushort offset = tpig.x+(tpig.y>0?cols*rows/2:0);// + tpisg);
    for (int r=0; r<rows/2; r+=16) {
        for (int i=0; i<16; i++) {
            out[i] += weights[(r+i)*cols + offset];//>0.02?weights[(r+i)*cols + offset]:weights[(r+i)*cols + offset];
        }
        

    }
    for (int i = 0; i<16; i++) { result[tpig.x] += out[i];};
}

/*
 working?
 
kernel void truthBucket2(device const half *weights [[buffer(0)]],
                       device half *result [[buffer(1)]],
                       constant ushort &rows [[buffer(2)]],
                       constant ushort &cols [[buffer(3)]],

                       ushort2 tpig [[thread_position_in_grid]]
                       ) {
    half out[16];
    for (int i = 0; i<16; i++) { out[i] = 0;};
    const ushort offset = tpig.x+(tpig.y>0?cols*rows/2:0);// + tpisg);
    for (int r=0; r<rows/2; r+=16) {
        for (int i=0; i<16; i++) {
            out[i] += weights[(r+i)*cols + offset];//>0.02?weights[(r+i)*cols + offset]:weights[(r+i)*cols + offset];
        }
        

    }
    for (int i = 0; i<16; i++) { result[tpig.x] += out[i];};
}
 
 */

kernel void bucketMul(
                   device const half *weights [[buffer(0)]],
                   device const float2 *dispatch [[buffer(1)]],
                   device half *result [[buffer(2)]],
                   constant ushort &rows [[buffer(3)]],
                   constant ushort &cols [[buffer(4)]],

                  ushort2 id [[thread_position_in_grid]]
                      ) {
    
    //
    // for groups == 1: go through every row of inDim, fetch its outCol value and += to myVal
    // for groups > 1 : go through rows groupId*groupSize to (groupId+1)*groupSize, fetch its outCol value and += to myVal
    

    float myVal[16];
                          //half z = 0;
      for (int i = 0; i<16; i++) { myVal[i] = 0;};
    
      const ushort rowOffset = id.y*65536/16;
      for (int r=0; r<65536/16; r+=1) {
          float2 d = dispatch[rowOffset + r];
          half w = weights[int(d[1])*cols + id.x];
          for (int i=0; i<16; i++) {
              //z=(as_type<ushort>(weights[int(d[1])*cols + id.x])&0xF);
              //z=weights[int(d[1])*cols + id.x];
              myVal[i] += (as_type<ushort>(w)&0x15) == i?d[0]*w:0;
          }
      }
      for (int i = 0; i<16; i++) { result[id.x*16+i] += myVal[i];};

    /*
    for (ushort i = 0; i<rows; i++) {
        half2 d = dispatch[rowOffset + i];
        int bucketPos = int(d[1]) + bucketCol;// * 86
        w = weights[bucketPos];
//        ushort binId = as_type<ushort>(w[wx][wy])&3;
        myVal[0] += d[0] * w[0][0];
    }
    
    const ushort outCol = id.x * 16 * 4 * 32;
    int off = 0;
    for (int j = 0; j<2; j++) {
        result[off+outCol+j] += myVal[j];//myVal[off+j];
    }*/
    
}



kernel void mul_col_4096(device const half *matrix [[buffer(0)]],
                    device const half *vector [[buffer(1)]],
                    device half *result [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    half sum = 0.0;
    int row = id;
    int offset = id * 4096;
    
    for (int i = 0; i < 4096; i++) {
        sum += matrix[(offset+i)] * vector[i];
//        sum += matrix[11008*i+id] * vector[i];

    }

    result[row] = sum;
}


kernel void mul_col_11008(device const half *matrix [[buffer(0)]],
                    device const half *vector [[buffer(1)]],
                    device half *result [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    half sum = 0.0;
    int row = id;
    int offset = id * 11008;
    
    for (int i = 0; i < 11008; i++) {
        sum += matrix[(offset+i)] * vector[i];
    }

    result[row] = sum;
}


#define outer_count 4096

kernel void internal(device const half *fxn [[buffer(0)]],
                     device const half *w1 [[buffer(1)]],
                     device const half *w3 [[buffer(2)]],
                     device half *result [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    half x1 = 0.0;
    half x2 = 0.0;
    half x3 = 0.0;
    int offset = id*outer_count;

    for (int i = 0; i<outer_count; i++) {
        half x = fxn[i];
        x1 += x * w1[offset + i];
        x3 += x * w3[offset + i];
    }
    x2 = x3 * x1 / (1 + exp(-x1));
    result[id] = x2;
}

kernel void second (device const half *matrix [[buffer(0)]],
                    device const half *vector [[buffer(1)]],
                    device half *result [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    half sum = 0.0;
    int offset = id * 11008;
    for (int i = 0; i < 11008; i++) {
        sum += matrix[(offset + i)] * vector[i];
    }

    result[id] += sum;
}

kernel void basicBitonicSort(device half     *floats     [[ buffer(0) ]],
                             constant int     &p             [[ buffer(1) ]],
                             constant int     &q             [[ buffer(2) ]],
                             uint             gid         [[ thread_position_in_grid ]])
{
    // taken from https://developer.apple.com/forums/thread/674181
    //            https://github.com/tgymnich/MetalSort

    int pMinusQ = p-q;
    int distance = 1 << pMinusQ;
    uint gidShiftedByP = gid >> p;
    // True: Increasing / False: Descreasing
    bool direction = (gidShiftedByP & 2) == 0;
    uint gidDistance = (gid & distance);
    bool isGidDistanceZero = (gidDistance == 0);
    uint gidPlusDistance = (gid | distance);
    bool isLowerIndexGreaterThanHigher = (floats[gid] > floats[gidPlusDistance]);
    if (isGidDistanceZero && isLowerIndexGreaterThanHigher == direction) {
        float temp = floats[gid];
        floats[gid] = floats[gidPlusDistance];
        floats[gidPlusDistance] = temp;
    }
}



/*
 
 VALID AND TESTED FAST
 kernel void bucketMul(device const half *vector [[buffer(0)]],
                    device const float4 *weights [[buffer(1)]],
                    device float *result [[buffer(2)]],
                    device const half *seedVec [[buffer(3)]],
                    constant int &inDim [[buffer(4)]],
                    constant int &outDim [[buffer(5)]],

                   uint id [[thread_position_in_grid]]) {
     
     //
     // for groups == 1: go through every row of inDim, fetch its outCol value and += to myVal
     // for groups > 1 : go through rows groupId*groupSize to (groupId+1)*groupSize, fetch its outCol value and += to myVal
     
     int groups = 32;
     int inDim4 = inDim/4;
     int rowsPerGroup = inDim4 / groups;
     int outCol = id / groups;
     int outGroup = id % groups;
     int rowBegin = outGroup * rowsPerGroup;
     int rowEnd = (outGroup+1) * rowsPerGroup;
     float myVal[16*4];
     int row = rowBegin;
     float4 w1;
     for (int i = 0; i<250; i+=1) { //
         w1 = weights[row+i]; //*(int(vector[i])&7)
         myVal[int(w1.x)&15] += vector[i] * (w1.x);
         myVal[int(w1.y)&15+16] += vector[i] * (w1.y);
         myVal[int(w1.z)&15+32] += vector[i] * (w1.z);
         myVal[int(w1.w)&15+46] += vector[i] * (w1.w);
     }
     for(int i = 0; i<4; i++) {
         int off = i*16;
         result[off+outCol] += myVal[off+0];
         result[off+outCol+1] += myVal[off+1];
         result[off+outCol+2] += myVal[off+2];
         result[off+outCol+3] += myVal[off+3];
         result[off+outCol+4] += myVal[off+4];
         result[off+outCol+5] += myVal[off+5];
         result[off+outCol+6] += myVal[off+6];
         result[off+outCol+7] += myVal[off+7];
         result[off+outCol+8] += myVal[off+8];
         result[off+outCol+9] += myVal[off+9];
         result[off+outCol+10] += myVal[off+10];
         result[off+outCol+11] += myVal[off+11];
         result[off+outCol+12] += myVal[off+12];
         result[off+outCol+13] += myVal[off+13];
         result[off+outCol+14] += myVal[off+14];
         result[off+outCol+15] += myVal[off+15];
     }

 //    atomic_fetch_add_explicit(&result[outCol], myVal, memory_order_relaxed);
     
 }
 
 
 */
